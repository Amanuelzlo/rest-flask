from flask import Flask, render_template
import flask
import joblib
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
import numpy as np


app = Flask(__name__)


# Use joblib to load in the pre-trained model.
with open(f'model/pima-trained-model.pkl', 'rb') as f:
    classifier = joblib.load(f)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=[ 'POST'])
def predict():
       if request.method == 'POST':
        
        Pregnancies = int(request.form['Pregnancies'])
        DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
        Age = int(request.form['Age'])
        Glucose = int(request.form['Glucose'])
        BloodPressure = int(request.form['BloodPressure'])
        SkinThickness = int(request.form['SkinThickness'])
        Insulin = int(request.form['Insulin'])
        BMI = float(request.form['BMI'])
        
        data = np.array([[Pregnancies, DiabetesPedigreeFunction, Age, Glucose, 
                          BloodPressure, SkinThickness, Insulin, BMI]])
        prediction = classifier.predict(data)
        
        return render_template('main.html', prediction=prediction)
                                     
if __name__ == '__main__':
    app.run(debug=True)