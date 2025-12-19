"""
Utility functions for TATA Steel Machine Failure Prediction
"""
import pandas as pd
import joblib
from src.logger import setup_logger

logger = setup_logger('utils')

def load_data(data_url):
    """
    Load dataset from URL
    
    Args:
        data_url (str): URL to dataset
    
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        logger.info("Loading dataset...")
        df = pd.read_csv(data_url)
        logger.info(f"Dataset loaded successfully: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def save_model(model, filepath):
    """
    Save a trained model using joblib
    
    Args:
        model: Trained model object
        filepath (str): Path to save the model
    """
    try:
        joblib.dump(model, filepath)
        logger.info(f"Model saved successfully to {filepath}")
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise

def load_model(filepath):
    """
    Load a trained model using joblib
    
    Args:
        filepath (str): Path to the saved model
    
    Returns:
        Loaded model object
    """
    try:
        model = joblib.load(filepath)
        logger.info(f"Model loaded successfully from {filepath}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def save_dataframe(df, filepath):
    """
    Save a DataFrame to CSV
    
    Args:
        df (pd.DataFrame): DataFrame to save
        filepath (str): Path to save the CSV
    """
    try:
        df.to_csv(filepath, index=False)
        logger.info(f"DataFrame saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving DataFrame: {str(e)}")
        raise

def load_dataframe(filepath):
    """
    Load a DataFrame from CSV
    
    Args:
        filepath (str): Path to the CSV file
    
    Returns:
        pd.DataFrame: Loaded DataFrame
    """
    try:
        df = pd.read_csv(filepath)
        logger.info(f"DataFrame loaded from {filepath}")
        return df
    except Exception as e:
        logger.error(f"Error loading DataFrame: {str(e)}")
        raise