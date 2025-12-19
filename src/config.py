"""
Configuration module for TATA Steel Machine Failure Prediction
Contains all constants, paths, and hyperparameters
"""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'saved_models')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

# Data URL
# Note: The dataset already contains the target column 'Machine failure'
DATA_URL = 'https://drive.google.com/uc?export=download&id=1Uq359_cmz-o-I2eEnUJpzPU500QdsH5U'

# Feature columns after preprocessing
FEATURES_TO_DROP = ['id', 'Product_ID', 'Type', 'Process_temperature_K_']
TARGET_COLUMN = 'Machine_failure'

# Final feature set for modeling
FINAL_FEATURES = [
    'Air_temperature_K_',
    'Rotational_speed_rpm_',
    'Torque_Nm_',
    'Tool_wear_min_',
    'TWF',
    'HDF',
    'PWF',
    'OSF',
    'RNF',
    'Type_encoded'
]

# Train-test split parameters
TEST_SIZE = 0.3
RANDOM_STATE = 42

# LightGBM Base Model Parameters
LGBM_BASE_PARAMS = {
    'class_weight': 'balanced',
    'random_state': RANDOM_STATE
}

# LightGBM Hyperparameter Tuning Grid
LGBM_PARAM_GRID = {
    'num_leaves': [15, 31, 63],
    'max_depth': [-1, 5, 10],
    'min_child_samples': [10, 20, 40],
    'min_child_weight': [1e-4, 1e-3, 1e-2],
    'class_weight': [None, 'balanced'],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 500],
    'reg_alpha': [0, 0.1, 0.4, 0.8],
    'reg_lambda': [0, 0.1, 0.4, 0.8]
}

# Model file paths
BASE_MODEL_PATH = os.path.join(MODEL_DIR, 'lgbm_base_model.pkl')
TUNED_MODEL_PATH = os.path.join(MODEL_DIR, 'lgbm_tuned_model.pkl')
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, 'preprocessor_config.pkl')

# Training parameters
CV_FOLDS = 3
SCORING_METRIC = 'f1'
N_ITER_RANDOM_SEARCH = 10  # Number of random search iterations

# Logging configuration
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = os.path.join(LOG_DIR, 'training.log')