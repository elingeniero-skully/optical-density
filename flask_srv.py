import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import numpy as np
import matplotlib.pyplot as plt
import calibration

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.secret_key = b'$RcxPi9J3PJ?s#Qr'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #Le fichier a été uploadé, on peut le traiter puis afficher la progressbar
            return process_file(filename)
    return render_template('mainpage.html')

def process_file(fname):
    
    # import image
    ima = plt.imread(os.path.join(app.config['UPLOAD_FOLDER'], fname))

    # profile
    m,n,_ = ima.shape #m représente la "hauteur", ordonnée de l'image.
    if m>n:
        profile = ima.mean(axis=1)
    else:
        profile = ima.mean(axis=0)

    # backgound
    bg = np.max(profile,axis=0)
    absorbance = - np.log10(profile/bg)

    #On veut récupérer le maximum de l'absorbance verte
    max_absorbance = np.max(absorbance, axis=0)[1]

    (model,x,y) = calibration.calibrate(('1.csv',0),('2.csv',2),('3.csv',5),('4.csv',10),('5.csv',15),('6.csv',20),('7.csv',35),('8.csv',50))
    estimated_concentration = calibration.get_prediction(model, max_absorbance)

    #Define a new scale for the progress bar
    estimated_progress = estimated_concentration/50*100 #same as doing *2

    #Si tout s'est bien passé
    return render_template('mainpage.html', filename=fname, progress=round(float(estimated_progress)), concentration=round(float(estimated_concentration),2))

    #Si une erreur est survenue
    #return render_template('mainpage.html', filename=name, error="Une erreur est survenue lors de l'interprétation de vos résultats.")
