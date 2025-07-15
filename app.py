from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load('model/pipeline.pkl')

# Dropdown options
CROP_OPTIONS = ['Rice', 'Wheat', 'Sugarcane', 'Maize', 'Cotton(lint)', 'Arecanut']
SEASON_OPTIONS = ['Kharif', 'Rabi', 'Whole Year']
STATE_OPTIONS = ['Tamil Nadu', 'Punjab', 'Maharashtra', 'Karnataka', 'Assam', 'Gujarat']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        crop = request.form['crop']
        season = request.form['season']
        state = request.form['state']
        area = float(request.form['area'])
        rainfall = float(request.form['rainfall'])
        fertilizer = float(request.form['fertilizer'])
        pesticide = float(request.form['pesticide'])

        fert_per_area = fertilizer / (area + 1e-6)
        pest_per_area = pesticide / (area + 1e-6)
        input_intensity = (fertilizer + pesticide) / (area + 1e-6)

        # Prepare input as DataFrame
        input_df = pd.DataFrame({
            'Crop': [crop],
            'Season': [season],
            'State': [state],
            'Area_log': [np.log1p(area)],
            'Annual_Rainfall': [rainfall],
            'Fertilizer_log': [np.log1p(fertilizer)],
            'Pesticide_log': [np.log1p(pesticide)],
            'Fertilizer_per_Area_log': [np.log1p(fert_per_area)],
            'Pesticide_per_Area_log': [np.log1p(pest_per_area)],
            'Input_Intensity_log': [np.log1p(input_intensity)]
        })

        # Prediction
        prediction = round(model.predict(input_df)[0], 2)

    return render_template('index.html',
                           prediction=prediction,
                           crop_options=CROP_OPTIONS,
                           season_options=SEASON_OPTIONS,
                           state_options=STATE_OPTIONS)

# No need for app.run() when using Gunicorn or Render
