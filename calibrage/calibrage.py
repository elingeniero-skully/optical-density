###THIS CODE IS NOT MINE, ALL CREDIT GOES TO GILLES BRUYLANTS <gilles.bruylants@ulb.be>

# importation des librairies 
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

counter = 0

def annot_max(x,y, ax=None,pos=(0.94,0.96)):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "x={:.3f}, y={:.3f}".format(xmax, ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=pos, **kw)

def select_callback(eclick, erelease):
    'eclick and erelease are the press and release events'
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
    print(" The button you used were: %s %s" % (eclick.button, erelease.button))
    process(x1,y1,x2,y2)

def process(x1,y1,x2,y2,name="default"):
    # image crop
    s = (slice(int(y1),int(y2)),slice(int(x1),int(x2)))
    crop = ima[s]
    ax2.clear()
    ax2.set_xlabel("cropped image")
    ax2.imshow(crop)

    # histogram
    bins = np.arange(-0.5, 255+1,1)
    hr = np.histogram(crop[:,:,0].flatten(), bins = bins)[0]
    hg = np.histogram(crop[:,:,1].flatten(), bins = bins)[0]
    hb = np.histogram(crop[:,:,2].flatten(), bins = bins)[0]
    ax3.clear()
    ax3.set_xlabel("rgb histogram")
    ax3.plot(bins[1:],hr,'r')
    ax3.plot(bins[1:],hg,'g')
    ax3.plot(bins[1:],hb,'b')
    
    # profile
    m,n,_ = crop.shape
    if m>n:
        profile = crop.mean(axis=1)
    else:
        profile = crop.mean(axis=0)

    # backgound
    bg = np.max(profile,axis=0)
    absorbance = - np.log10(profile/bg)

    # save profile to .csv
    global counter
    counter += 1

    np.savetxt(f'{str(9-counter)}.csv', absorbance,delimiter=',')

    print(bg)
    print(profile.shape)
    ax4.clear()
    ax4.set_xlabel("profile")
    ax4.plot(absorbance[:,0],c='r')
    ax4.plot(absorbance[:,1],c='g')
    ax4.plot(absorbance[:,2],c='b')

    annot_max(np.arange(absorbance.shape[0]),absorbance[:,0],ax=ax4, pos=(0.94,0.96))
    annot_max(np.arange(absorbance.shape[0]),absorbance[:,1],ax=ax4, pos=(0.30,0.90))
    annot_max(np.arange(absorbance.shape[0]),absorbance[:,2],ax=ax4, pos=(0.30,0.60))
    ax4.set_ylim([0,1])
    plt.draw()



# MAIN

jpg_filenames = [n for n in os.listdir() if n.endswith('.jpg')]
print(f'images: {jpg_filenames}')

for fname in jpg_filenames:
    fig, ((ax1, ax2), (ax3, ax4))  = plt.subplots(2,2,squeeze=True)
    plt.subplots_adjust(bottom=0.1,left=.05,right=0.99, top=1)
    # import image
    ima = plt.imread(fname)
    print(80*'*')
    print(f'processing {fname}')
    print(80*'*')
    print(f'size: {ima.shape}')
    # display
    
    ax1.set_xlabel(f'input image: {fname}')
    ax1.imshow(ima)
    # select rectangle
    selector = RectangleSelector(ax1, select_callback,
                                       drawtype='box', useblit=True,
                                       button=[1, 3],  # don't use middle button
                                       minspanx=5, minspany=5,
                                       spancoords='pixels',
                                       interactive=True)

    plt.show()

