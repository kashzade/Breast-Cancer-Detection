# load libraries
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('breast_cancer_detector.pickle', 'rb')) # Load ML model

# Exicute the HTML page
@app.route('/')
def home():
    return render_template('index.html') 

# Fetch input value from HTML form and show the predicted output on HTML page
@app.route('/predict',methods=['POST']) 
def predict():
    input_features = [float(x) for x in request.form.values()] # Take features value from HTML form
    features_value = [np.array(input_features)]
    
    features_name = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
       'mean smoothness', 'mean compactness', 'mean concavity',
       'mean concave points', 'mean symmetry', 'mean fractal dimension',
       'radius error', 'texture error', 'perimeter error', 'area error',
       'smoothness error', 'compactness error', 'concavity error',
       'concave points error', 'symmetry error', 'fractal dimension error',
       'worst radius', 'worst texture', 'worst perimeter', 'worst area',
       'worst smoothness', 'worst compactness', 'worst concavity',
       'worst concave points', 'worst symmetry', 'worst fractal dimension']
    
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df) # Predict cancer using ML model
        
    if output == 1:
        res_val = "** breast cancer **"
    else:
        res_val = "no breast cancer"
        

    return render_template('index.html', prediction_text='Patient has {}'.format(res_val))

# Exicute the app
if __name__ == "__main__":
    app.run()