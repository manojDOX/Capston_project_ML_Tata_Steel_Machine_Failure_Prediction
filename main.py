"""
Main pipeline for TATA Steel Machine Failure Prediction
Orchestrates the entire training workflow:
1. Data Loading
2. Preprocessing
3. Model Training (Base + Tuned)
4. Model Testing
"""
import sys
import argparse
from datetime import datetime
from src.logger import setup_logger
from src.config import DATA_URL
from src.utils import load_data
from src.preprocessing import run_preprocessing
from src.train import run_training
from src.test import run_testing

# Setup logger
logger = setup_logger('main_pipeline')

def main(skip_data_load=False, skip_training=False, test_only=False):
    """
    Execute the complete ML pipeline
    
    Args:
        skip_data_load (bool): Skip data loading if already processed
        skip_training (bool): Skip training if models already exist
        test_only (bool): Only run testing on existing model
    """
    start_time = datetime.now()
    logger.info("=" * 70)
    logger.info("TATA STEEL MACHINE FAILURE PREDICTION - ML PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Pipeline started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # ===================================================================
        # STEP 1: DATA LOADING
        # ===================================================================
        if not skip_data_load:
            logger.info("\n" + "=" * 70)
            logger.info("STEP 1: DATA LOADING")
            logger.info("=" * 70)
            
            # Load the complete dataset (includes target column)
            df = load_data(DATA_URL)
            
            logger.info(f"Complete dataset shape: {df.shape}")
            logger.info(f"Dataset contains target column 'Machine failure': {'Machine failure' in df.columns}")
        
        # ===================================================================
        # STEP 2: PREPROCESSING
        # ===================================================================
        if not skip_data_load:
            logger.info("\n" + "=" * 70)
            logger.info("STEP 2: DATA PREPROCESSING")
            logger.info("=" * 70)
            
            X_train, X_test, y_train, y_test = run_preprocessing(df)
        else:
            logger.info("\n" + "=" * 70)
            logger.info("STEP 2: LOADING PREPROCESSED DATA")
            logger.info("=" * 70)
            
            from src.config import PROCESSED_DATA_DIR
            import pandas as pd
            import os
            
            X_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_train.csv'))
            X_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_test.csv'))
            y_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'y_train.csv')).squeeze()
            y_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'y_test.csv')).squeeze()
            
            logger.info(f"Loaded preprocessed data:")
            logger.info(f"  X_train: {X_train.shape}")
            logger.info(f"  X_test: {X_test.shape}")
        
        # ===================================================================
        # STEP 3: MODEL TRAINING
        # ===================================================================
        if not test_only and not skip_training:
            logger.info("\n" + "=" * 70)
            logger.info("STEP 3: MODEL TRAINING")
            logger.info("=" * 70)
            
            base_model, tuned_model, base_metrics, tuned_metrics = run_training(
                X_train, y_train, X_test, y_test
            )
        
        # ===================================================================
        # STEP 4: MODEL TESTING
        # ===================================================================
        logger.info("\n" + "=" * 70)
        logger.info("STEP 4: COMPREHENSIVE MODEL TESTING")
        logger.info("=" * 70)
        
        # Test the tuned model (or base model if tuning was skipped)
        from src.config import TUNED_MODEL_PATH
        test_results = run_testing(X_test, y_test, TUNED_MODEL_PATH, generate_plots=True)
        
        # ===================================================================
        # PIPELINE COMPLETION
        # ===================================================================
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("\n" + "=" * 70)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        logger.info(f"Total execution time: {duration}")
        logger.info(f"Pipeline ended at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        logger.info("\n" + "=" * 70)
        logger.info("FINAL RESULTS SUMMARY")
        logger.info("=" * 70)
        logger.info(f"ROC-AUC Score: {test_results['roc_auc_score']:.4f}")
        logger.info(f"Precision-Recall AUC: {test_results['precision_recall_auc']:.4f}")
        logger.info(f"Test Recall (Class 1): {test_results['classification_report']['1']['recall']:.4f}")
        logger.info(f"Test Precision (Class 1): {test_results['classification_report']['1']['precision']:.4f}")
        logger.info(f"Test F1-Score (Class 1): {test_results['classification_report']['1']['f1-score']:.4f}")
        
        logger.info("\n[SUCCESS] All models and artifacts have been saved successfully")
        logger.info(f"[INFO] Check the 'models/saved_models/' directory for trained models")
        logger.info(f"[INFO] Check the 'logs/plots/' directory for visualization plots")
        
        return {
            'success': True,
            'test_results': test_results,
            'duration': duration
        }
        
    except Exception as e:
        logger.error(f"\n[ERROR] Pipeline failed with error: {str(e)}")
        logger.exception("Full traceback:")
        return {
            'success': False,
            'error': str(e)
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TATA Steel ML Pipeline')
    parser.add_argument('--skip-data-load', action='store_true',
                       help='Skip data loading and use preprocessed data')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip model training')
    parser.add_argument('--test-only', action='store_true',
                       help='Only run testing on existing model')
    
    args = parser.parse_args()
    
    result = main(
        skip_data_load=args.skip_data_load,
        skip_training=args.skip_training,
        test_only=args.test_only
    )
    
    # Exit with appropriate code
    sys.exit(0 if result['success'] else 1)