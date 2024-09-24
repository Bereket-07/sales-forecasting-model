from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # Import CORS
import pandas as pd
import numpy as np
import joblib
import gdown
import os
import sys

# Ensure your custom pipeline script is in the path
sys.path.append(os.path.abspath('../../scripts/model_training'))
import pipeline_withotgrid as p  # Ensure this path is correct

app = Flask(__name__)
CORS(app)  # Enable CORS for the app

# Google Drive URL and output path for the model file
model_url = 'https://drive.google.com/uc?export=download&id=1oIVpESdt2JpQDv3qTkdG0NCQRQCx-dG1'
model_output_path = "final_model.pkl"

# Function to download the model from Google Drive if not already downloaded
def download_model():
    if not os.path.exists(model_output_path):
        print("Downloading model from Google Drive...")
        gdown.download(model_url, model_output_path, quiet=False)

# Download and load the model when the app starts
download_model()
pipeline = joblib.load(model_output_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve data from the request
        data = request.json
        store_id = data['store_id']
        day_of_week = data['day_of_week']
        date = data['date']
        open_store = data['open_store']
        promo = data['promo']
        state_holiday = data['state_holiday']
        school_holiday = data['school_holiday']

        # Create a DataFrame for the input data
        input_data = pd.DataFrame({
            'Store': [store_id],
            'DayOfWeek': [day_of_week],
            'Date': [date],  # Ensure the model can handle this format
            'Open': [open_store],
            'Promo': [promo],
            'StateHoliday': [state_holiday],
            'SchoolHoliday': [school_holiday]
        })

        # Apply any necessary preprocessing on the 'Date' field (if your model needs this)
        input_data['Date'] = pd.to_datetime(input_data['Date'])

        # Make prediction
        predicted_sales = pipeline.predict(input_data)

        # Return prediction result
        return jsonify({'store_id': store_id, 'predicted_sales': predicted_sales[0]})

    except Exception as e:
        # Handle errors and log them
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
