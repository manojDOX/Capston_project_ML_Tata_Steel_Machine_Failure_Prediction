"""
Data preprocessing module for TATA Steel Machine Failure Prediction
Handles feature engineering and data preparation
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.logger import setup_logger
from src.config import (
    TARGET_COLUMN, FINAL_FEATURES, TEST_SIZE, 
    RANDOM_STATE, PROCESSED_DATA_DIR
)
import os

logger = setup_logger('preprocessing')

class DataPreprocessor:
    """
    Class to handle all data preprocessing steps
    """
    
    def __init__(self):
        self.feature_columns = None
        self.target_column = TARGET_COLUMN
        
    def clean_column_names(self, df):
        """
        Clean column names by removing special characters
        
        Args:
            df (pd.DataFrame): Input DataFrame
        
        Returns:
            pd.DataFrame: DataFrame with cleaned column names
        """
        logger.info("Cleaning column names...")
        df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)
        logger.info("Column names cleaned successfully")
        return df
    
    def handle_missing_values(self, df):
        """
        Handle missing values in the dataset
        
        Args:
            df (pd.DataFrame): Input DataFrame
        
        Returns:
            pd.DataFrame: DataFrame with handled missing values
        """
        logger.info("Checking for missing values...")
        missing_count = df.isnull().sum().sum()
        
        if missing_count == 0:
            logger.info("No missing values found")
        else:
            logger.info(f"Found {missing_count} missing values")
            # Implement imputation logic here if needed
            
        return df
    
    def encode_categorical_features(self, df):
        """
        Encode categorical features (Type column)
        
        Args:
            df (pd.DataFrame): Input DataFrame
        
        Returns:
            pd.DataFrame: DataFrame with encoded features
        """
        logger.info("Encoding categorical features...")
        
        if 'Type' in df.columns:
            df['Type_encoded'] = df['Type'].map({'L': 1, 'M': 2, 'H': 3})
            logger.info("Type column encoded: L=1, M=2, H=3")
        
        return df
    
    def select_features(self, df):
        """
        Select final features for modeling
        
        Args:
            df (pd.DataFrame): Input DataFrame
        
        Returns:
            pd.DataFrame: DataFrame with selected features
        """
        logger.info("Selecting final features for modeling...")
        
        # Ensure target column is included
        columns_to_keep = FINAL_FEATURES + [self.target_column]
        
        # Check if all required columns exist
        missing_cols = [col for col in columns_to_keep if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing columns: {missing_cols}")
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        df_final = df[columns_to_keep].copy()
        logger.info(f"Final dataset shape: {df_final.shape}")
        
        return df_final
    
    def preprocess(self, df):
        """
        Execute full preprocessing pipeline
        
        Args:
            df (pd.DataFrame): Raw input DataFrame
        
        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        logger.info("Starting preprocessing pipeline...")
        
        # Step 1: Clean column names
        df = self.clean_column_names(df)
        
        # Step 2: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 3: Encode categorical features
        df = self.encode_categorical_features(df)
        
        # Step 4: Select final features
        df = self.select_features(df)
        
        logger.info("Preprocessing pipeline completed successfully")
        return df
    
    def split_data(self, df, test_size=TEST_SIZE, random_state=RANDOM_STATE):
        """
        Split data into train and test sets
        
        Args:
            df (pd.DataFrame): Preprocessed DataFrame
            test_size (float): Proportion of test set
            random_state (int): Random seed for reproducibility
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        logger.info("Splitting data into train and test sets...")
        
        # Separate features and target
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        # Split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            stratify=y, 
            random_state=random_state
        )
        
        logger.info(f"Training set shape: X={X_train.shape}, y={y_train.shape}")
        logger.info(f"Test set shape: X={X_test.shape}, y={y_test.shape}")
        logger.info(f"Class distribution in train: {y_train.value_counts().to_dict()}")
        logger.info(f"Class distribution in test: {y_test.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def save_processed_data(self, X_train, X_test, y_train, y_test):
        """
        Save processed data splits to CSV files
        
        Args:
            X_train, X_test, y_train, y_test: Data splits
        """
        logger.info("Saving processed data...")
        
        X_train.to_csv(os.path.join(PROCESSED_DATA_DIR, 'X_train.csv'), index=False)
        X_test.to_csv(os.path.join(PROCESSED_DATA_DIR, 'X_test.csv'), index=False)
        y_train.to_csv(os.path.join(PROCESSED_DATA_DIR, 'y_train.csv'), index=False)
        y_test.to_csv(os.path.join(PROCESSED_DATA_DIR, 'y_test.csv'), index=False)
        
        logger.info("Processed data saved successfully")

def run_preprocessing(df_features):
    """
    Execute the preprocessing pipeline
    
    Args:
        df_features (pd.DataFrame): Raw feature DataFrame
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    preprocessor = DataPreprocessor()
    
    # Preprocess the data
    df_processed = preprocessor.preprocess(df_features)
    
    # Split the data
    X_train, X_test, y_train, y_test = preprocessor.split_data(df_processed)
    
    # Save processed data
    preprocessor.save_processed_data(X_train, X_test, y_train, y_test)
    
    return X_train, X_test, y_train, y_test