import pandas as pd
from config import Config
from logs.logger import setup_logger

logger = setup_logger("data_loader_logger", "logs/data_loader.log")

def load_data():
    try:
        data = pd.read_csv(Config.DATA_PATH)
        logger.info(f"Data loaded successfully from {Config.DATA_PATH}")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise FileNotFoundError(f"Unable to load data from {Config.DATA_PATH}")
