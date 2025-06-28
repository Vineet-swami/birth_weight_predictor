from flask import Flask, jsonify, request, render_template
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)

def get_cleaned_data(form_data):
    gestation = float(form_data['gestation'])
    parity = float(form_data['parity'])
    age = float(form_data['age'])   
    height = float(form_data['height'])
    weight = float(form_data['weight'])
    smoke = str(form_data['smoke'])

    return {
        'gestation': [gestation],
        'parity': [parity],
        'age': [age],
        'height': [height],
        'weight': [weight],
        'smoke': [1 if smoke == 'yes' else 0]
    }

@app.route('/', methods=['GET'])
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def get_prediction():
    baby_data = request.form

    cleaned_baby_data = get_cleaned_data(baby_data)
    baby_df = pd.DataFrame(cleaned_baby_data)
    with open('model/linear_regression_model.pkl', 'rb') as obj:
        model = pickle.load(obj)
    prediction = model.predict(baby_df)

    prediction= round(float(prediction), 2)
    response = { "prediction": prediction }
    return render_template("index.html", prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)