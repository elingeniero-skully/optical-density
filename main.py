import os
from flask import Flask, request, url_for, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt, mpld3
import csv
from sklearn.linear_model import LinearRegression

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

#Configuration
app = Flask(__name__)
app.secret_key = b'$RcxPi9J3PJ?s#Qr'
app.config['UPLOAD_FOLDER'] = 'uploads'

#Really useful to prevent problems with symbolic links
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
uploads_dir = os.path.join(__location__, app.config['UPLOAD_FOLDER'])

#### DEBUT FONCTIONS POUR GENERER LA DROITE DE CALIBRATION

def compute_max_green_absorbance(csv_filepath):
    """Return the mean green absorbance value from a .csv file"""
    with open(csv_filepath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        absorbance = 0
        for row in csv_reader:
            if float(row[1]) > absorbance:
                absorbance = float(row[1])
    return absorbance

def compute_max_green_blue_absorbance(csv_filepath):
    """Return the mean green absorbance value from a .csv file"""
    with open(csv_filepath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        absorbance = 0
        for row in csv_reader:
            if float(row[1]) > absorbance:
                absorbance = float(row[1])
            if float(row[2]) > absorbance:
                absorbance = float(row[2])
    return absorbance

def compute_mean_green_absorbance(csv_filepath):
    """Return the mean green absorbance value from a .csv file"""
    with open(csv_filepath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        absorbance = 0
        for row in csv_reader:
            absorbance += float(row[1])
            line_count += 1
        absorbance /= line_count
    return absorbance

def calibrate(args):
    """compute the regression line based on csv files"""
    #arguments de la forme ("file.csv", concentration AuNP)
    absorbances = []
    concentrations = []

    for arg in args: #for each tuple
        absorbances.append(compute_max_green_absorbance(os.path.join("uploads", arg[0])))
        concentrations.append(arg[1])

    #absorbances.insert(0,0)
    #concentrations.insert(0,0)

    x,y = np.array(absorbances).reshape(-1, 1), np.array(concentrations)
    model = LinearRegression().fit(x, y)

    #r_sq = model.score(x, y)
    return model, x, y

def generate_web_calibration_graph(model,x,y,sample_x):
    """Returns a html string corresponding to a matplotlib figure that is visible in a web browser."""
    #print(f"coefficient of determination: {r_sq}")
    #print(f"intercept: {model.intercept_}")
    #print(f"slope: {model.coef_}")

    #sample is a tuple.

    x_ = np.linspace(0, max(x), num=25) #x represents the absorbance, thus the values are comprised between 0 and 1.
    fig, ax = plt.subplots(nrows=1) #Generate 1 subplot on which we will "draw" the scatters (dots) and the regression line.
    ax.scatter(x,y, color='g') #Dots
    ax.plot(x_, model.predict(x_.reshape(-1,1)), color='k') #Regression line
    ax.set_title("Droite de régression")
    ax.set_xlabel("Absorbance mesurée")
    ax.set_ylabel("Concentration en ovalbumine estimée (mol/l)")
    ax.set_xlim([0, max(x)])

    if sample_x:
        sample_y = model.predict(np.array(sample_x).reshape(-1,1))
        ax.set_ylim([0, max(max(y),sample_y)])
        ax.scatter(sample_x, sample_y, color='r')
    else:
        ax.set_ylim([0, max(y)])

    html_fig_string = mpld3.fig_to_html(fig, d3_url=None, mpld3_url=None, no_extras=False, template_type='simple', figid=None, use_http=False, include_libraries=True)
    return html_fig_string

#### FIN FONCTIONS POUR GENERER LA DROITE DE CALIBRATION

#### FIN FONCTIONS ECHANTILLON

def csv_from_jpg(filepath, filename, is_sample=False):
    """Génère un profil d'absorbance pour l'image spécifiée en entrée, et sauvegarde au format .csv"""

    ima = np.asarray(Image.open(filepath))
    height,width,depth = ima.shape

    #On génère le profil d'absorbance
    if height > width:
        profile = ima.mean(axis=1) #Si l'image est plus haute que large, alors on génère le profil d'absorbance selon la largeur. (On "réduit" sa largeur à zéro)
    else:
        profile = ima.mean(axis=0) #Si l'image est plus large que haute, alors on génère le profil d'absorbance selon la hauteur.

    #On génère le profil du "bruit de fond"
    #Le calcul est acceptable en considérant que la zone rouge est petite comparée à la zone totale de la bandelette.
    #Note : on prend toujours axis=0 pour background car peu importe l'axe sur lequel on calcule le profil, le profil a toujours les mêmes dimensions.

    bg = np.max(profile,axis=0)
    absorbance = - np.log10(profile/bg)

    #Je vois le rouge, cela signifie que le bleu et le vert ont été absorbés (car on envoie une lumière blanche).

    #On enregistre le fichier au format texte (.csv), qui contient le profil d'absorbance pour chaque composante de couleur (R,G,B)
    
    if is_sample:
        fp = os.path.join(uploads_dir, 'sample', ''.join([filename, '.csv']))
    else:
        fp = os.path.join(uploads_dir, ''.join([filename, '.csv']))
    np.savetxt(fp, absorbance, delimiter=',')
    return fp

    #np.savetxt(f'.\\uploads\\{filename.split(".")[0]}.csv', absorbance, delimiter=',')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#Génération de la page d'accueil
@app.route('/')
def main():
    return render_template('main.html', view="homepage")

def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

#Génération de la page pour générer une nouvelle droite de calibration
@app.route('/new_calibration', methods=['GET','POST'])
def new_calibration():
    #Traiter l'envoi du formulaire
    if request.method == "POST":
        #On enregistre les fichiers
        files = request.files.getlist("calibration-files") 

        #On supprime l'ancienne calibration
        if files[0].filename:
            items = os.listdir(uploads_dir)
            if items:
                for item in items: #checker le cas où le dossier est vide
                    if item != 'sample':
                        os.remove(os.path.join(uploads_dir, item))

        for file in files:
            file_prefix = '.'.join((file.filename).split('.')[:-1])
            if is_float(file_prefix):
                #On définit le nouveau nom comme la nouvelle concentration (en notation scientifique)
                concentration = float(file_prefix) #concentration de référence / dilution
                savepath = os.path.join(uploads_dir, secure_filename(str(concentration)))
                file.save(savepath)
                csv_from_jpg(savepath, secure_filename(str(concentration)))
            else:
                print("Le fichier " + file.filename + " n'est pas un coefficient de dilution valide.")
                return render_template('main.html', view="new_calibration")

        return render_template('main.html', view="homepage")
    else:
        return render_template('main.html', view="new_calibration")

#Génération de la page pour visualiser (et générer à partir des csv) la droite de calibration actuelle
@app.route('/current_calibration')
def current_calibration():
    #On génère la droite de calibration
    calibration_files = []
    for calibration_file in os.listdir(app.config['UPLOAD_FOLDER']):
        if calibration_file.split('.')[-1] == "csv":
            calibration_files.append((calibration_file, float('.'.join(calibration_file.split('.')[:-1])))) #le float va poser probleme a cause de la virgule (c'est réglé)
    if calibration_files:
        (model,x,y) = calibrate(calibration_files)
        calibration_graph = generate_web_calibration_graph(model,x,y,0)
        return render_template('main.html', view="current_calibration", calibration_line = calibration_graph)
    else:
        return render_template('main.html', view="current_calibration")

#Génération de la page pour réaliser un nouveau test d'échantillon.
@app.route('/test_sample', methods=['GET','POST'])
def test_sample():
    if request.method == "POST":
        photo_sample_file = request.files['photo-sample-file']
        if photo_sample_file and allowed_file(photo_sample_file.filename):
            filename = secure_filename(''.join(photo_sample_file.filename.split('.')[:-1]))
            photo_sample_file.save(os.path.join(uploads_dir, 'sample', filename)) #Sauvegarde du fichier échantillon
            #On génère la droite de calibration 
            path_to_csv_file = csv_from_jpg(os.path.join(uploads_dir, 'sample', filename), filename, is_sample=True)
            calibration_files = []
            for calibration_file in os.listdir(app.config['UPLOAD_FOLDER']):
                if calibration_file.split('.')[-1] == "csv":
                    calibration_files.append((calibration_file, float(''.join(calibration_file.split('.')[:-1])))) #le float va poser probleme a cause de la virgule (c'est réglé)
            if calibration_files:
                (model,x,y) = calibrate(calibration_files)
                #On ajoute notre point
                sample_x = compute_max_green_absorbance(path_to_csv_file)
                #On supprime le fichier csv
                os.remove(path_to_csv_file)
                os.remove(os.path.join(uploads_dir, 'sample', filename))
                #On calcule la concentration grâce au modèle et on l'ajoute sur la droite de calibration
                calibration_graph = generate_web_calibration_graph(model,x,y,sample_x)
                return render_template('main.html', view="current_calibration", calibration_line = calibration_graph)
            else:
                return render_template('main.html', view="test_sample")

            # ainsi que le point (jouer sur la moyenne de l'absorbance ?)
    return render_template('main.html', view="test_sample")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
