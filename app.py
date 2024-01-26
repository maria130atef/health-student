from flask import Flask, render_template, request
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pickle

app = Flask(__name__)

# Load the trained decision tree classifier
with open('dt_classifier.pkl', 'rb') as f:
    loaded_dt_classifier = pickle.load(f)

@app.route('/')
def index():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    weight = float(request.form['weight'])
    length = float(request.form['length'])
    hemoglobin = float(request.form['hemoglobin'])

    # Prepare input features for prediction
    input_features = np.array([[length, weight, hemoglobin]])

    # Make prediction using the loaded model
    prediction = loaded_dt_classifier.predict(input_features)

    # Determine the prediction result
    if prediction[0] == 'anemia':
        result = 'The patient is predicted to have anemia.'
    else:
        result = 'The patient is predicted to be normal.'

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
