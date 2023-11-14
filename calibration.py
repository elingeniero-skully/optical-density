import csv
import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

def compute_max_green_absorbance(csv_filepath):
    """Return the mean green absorbance value from a .csv file"""
    with open(csv_filepath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        absorbance = 0
        for row in csv_reader:
            if float(row[1]) > absorbance:
                absorbance = float(row[1])

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

def calibrate(*args):
    """compute the regression line based on csv files"""
    #arguments de la forme ("file.csv", concentration AuNP)
    absorbances = []
    concentrations = []

    for arg in args: #for each tuple
        absorbances.append(compute_max_green_absorbance(arg[0]))
        concentrations.append(arg[1])

    #absorbances.insert(0,0)
    #concentrations.insert(0,0)

    x = np.array(absorbances).reshape(-1, 1)
    y = np.array(concentrations)

    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)
    
    return model, x, y

def get_prediction(model, X):
    """Accepts a trained model and a value x, returns the estimate value of y"""
    return model.predict(X.reshape(-1,1))

def draw_calibration_graph(model,x,y):
    #print(f"coefficient of determination: {r_sq}")
    #print(f"intercept: {model.intercept_}")
    #print(f"slope: {model.coef_}")
    x_ = np.linspace(0, 0.1)
    plt.scatter(x, y, color='g')
    plt.plot(x_, model.predict(x_.reshape(-1,1)), color='k')
    plt.show()

if __name__ == "__main__":
    (model,x,y) = calibrate(('1.csv',0),('2.csv',2),('3.csv',5),('4.csv',10),('5.csv',15),('6.csv',20),('7.csv',35),('8.csv',50))
    draw_calibration_graph(model,x,y)
    print(model.score(x,y))
