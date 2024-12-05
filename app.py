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
        # Load dataset to identify categorical columns dynamically
        df = pd.read_csv(config.DATA_PATH)
        categorical_columns = identify_categorical_columns(df)

        # Extract features from the form
        features = []
        for i in range(1, len(df.columns)):  # Dynamically match the form fields to the dataset columns
            form_field = request.form.get(f"feature_{i}")
            if form_field is not None:
                # Check if the feature is categorical or numeric
                if df.columns[i - 1] in categorical_columns:
                    # If categorical, apply label encoding
                    features.append(label_encoder.transform([form_field])[0])
                else:
                    # If numeric, convert to float and append
                    features.append(float(form_field))

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
