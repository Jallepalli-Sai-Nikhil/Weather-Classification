from flask import Flask, render_template, request
import joblib
import logging
import pandas as pd
from logs.logger import setup_logger
from config import Config
from exceptions.custom_exceptions import ModelLoadError
from utils.data_loader import load_data
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Initialize Flask app
app = Flask(__name__)

# Setup logger
logger = setup_logger("app_logger", "logs/app.log")

# Load configurations
config = Config()

# Load pre-trained model, preprocessor, and label encoder
try:
    model = joblib.load(config.MODEL_PATH)
    preprocessor = joblib.load(config.PREPROCESSOR_PATH)
    label_encoder = joblib.load(config.LABEL_ENCODER_PATH)  # Load the label encoder
    logger.info("Model, preprocessor, and label encoder loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model, preprocessor, or label encoder: {str(e)}")
    raise ModelLoadError(f"Failed to load model, preprocessor, or label encoder: {e}")

# Identify categorical columns dynamically from the dataset
def identify_categorical_columns(data):
    categorical_columns = data.select_dtypes(include=["object"]).columns.tolist()
    return categorical_columns

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Manually define the expected features
        expected_features = [
            'Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)', 
            'Cloud Cover', 'Atmospheric Pressure', 'UV Index', 'Season', 
            'Visibility (km)', 'Location', 'Weather Type'
        ]

        # Extract features from the form (manual mapping to each feature)
        features = []
        
        # Loop through each feature and get the corresponding form value
        for i, feature in enumerate(expected_features):
            form_field = request.form.get(f"feature_{i + 1}")  # Form field starts from 1
            if form_field is not None:
                # Check if the feature is categorical or numeric
                if feature in ['Season', 'Location', 'Weather Type', 'Cloud Cover']:  # These are categorical features
                    # Apply label encoding to categorical features
                    encoded_value = label_encoder.transform([form_field])[0]
                    features.append(encoded_value)  # Add encoded value
                else:
                    # For numeric features, convert to float and append
                    features.append(float(form_field))

        # Ensure all features have been provided
        if len(features) != len(expected_features):
            raise ValueError("Not all required features were provided in the form.")

        # Apply preprocessing (scaling) on the numeric features
        processed_features = preprocessor.transform([features])

        # Predict using the trained model
        prediction = model.predict(processed_features)

        # Log the prediction
        logger.info(f"Prediction made: {prediction}")
        return render_template("result.html", prediction=prediction[0])
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return render_template("error.html", error_message="Prediction failed. Please try again.")



if __name__ == "__main__":
    app.run(debug=True)
