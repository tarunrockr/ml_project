import numpy as np
import pandas as pd
from flask import Flask, request, render_template

from sklearn.preprocessing import StandardScaler

from src.pipeline.predict_pipeline import PredictPipeline, CustomData

app =  Flask(__name__)

## Routes

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Prediction Page
@app.route('/predictions', methods=['GET','POST'])
def predict_output():
    if request.method == 'GET':
        return render_template('prediction.html')
    if request.method == 'POST':
        data = CustomData(
            gender                      = request.form.get('gender'),
            race_ethnicity              = request.form.get('race_ethnicity'),
            parental_level_of_education = request.form.get('parental_level_of_education'),
            lunch                       = request.form.get('lunch'),
            test_preparation_course     = request.form.get('test_preparation_course'),
            reading_score               = request.form.get('reading_score'),
            writing_score               = request.form.get('writing_score')
        )

        input_dataframe = data.convert_data_in_dataframe()
        print("Input Dataframe: \n", input_dataframe)

        prediction_obj = PredictPipeline(input_dataframe)
        prediction = prediction_obj.predict()

        return render_template('prediction.html', results = prediction)



if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)