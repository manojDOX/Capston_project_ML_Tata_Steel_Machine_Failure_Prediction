"""
Model training module for TATA Steel Machine Failure Prediction
Trains both base and hyperparameter-tuned LightGBM models
"""
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from src.logger import setup_logger
from src.config import (
    LGBM_BASE_PARAMS, LGBM_PARAM_GRID, 
    BASE_MODEL_PATH, TUNED_MODEL_PATH,
    CV_FOLDS, SCORING_METRIC, N_ITER_RANDOM_SEARCH,
    RANDOM_STATE
)
from src.utils import save_model

logger = setup_logger('training')

class ModelTrainer:
    """
    Class to handle model training operations
    """
    
    def __init__(self):
        self.base_model = None
        self.tuned_model = None
        self.best_params = None
        
    def train_base_model(self, X_train, y_train):
        """
        Train base LightGBM model with default parameters
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
        
        Returns:
            LGBMClassifier: Trained base model
        """
        logger.info("=" * 50)
        logger.info("Training Base LightGBM Model")
        logger.info("=" * 50)
        
        try:
            # Initialize base model
            self.base_model = LGBMClassifier(**LGBM_BASE_PARAMS)
            
            # Train the model
            logger.info("Fitting base model...")
            self.base_model.fit(X_train, y_train)
            
            logger.info("Base model training completed successfully")
            
            # Save the base model
            save_model(self.base_model, BASE_MODEL_PATH)
            
            return self.base_model
            
        except Exception as e:
            logger.error(f"Error training base model: {str(e)}")
            raise
    
    def evaluate_model(self, model, X_train, y_train, X_test, y_test, model_name="Model"):
        """
        Evaluate model performance on train and test sets
        
        Args:
            model: Trained model
            X_train, y_train: Training data
            X_test, y_test: Test data
            model_name (str): Name of the model for logging
        """
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Evaluating {model_name}")
        logger.info(f"{'=' * 50}")
        
        try:
            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Training metrics
            logger.info("\n--- Training Set Performance ---")
            train_report = classification_report(y_train, y_train_pred, output_dict=True)
            logger.info(f"\n{classification_report(y_train, y_train_pred)}")
            
            # Test metrics
            logger.info("\n--- Test Set Performance ---")
            test_report = classification_report(y_test, y_test_pred, output_dict=True)
            logger.info(f"\n{classification_report(y_test, y_test_pred)}")
            
            # Key metrics for class 1 (failure)
            logger.info("\n--- Key Metrics for Machine Failure (Class 1) ---")
            logger.info(f"Test Recall: {test_report['1']['recall']:.4f}")
            logger.info(f"Test Precision: {test_report['1']['precision']:.4f}")
            logger.info(f"Test F1-Score: {test_report['1']['f1-score']:.4f}")
            
            # Confusion matrix
            cm_test = confusion_matrix(y_test, y_test_pred)
            logger.info(f"\nTest Confusion Matrix:\n{cm_test}")
            
            return {
                'train_report': train_report,
                'test_report': test_report,
                'confusion_matrix': cm_test
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise
    
    def tune_model(self, X_train, y_train):
        """
        Perform hyperparameter tuning using RandomizedSearchCV
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
        
        Returns:
            LGBMClassifier: Best tuned model
        """
        logger.info("=" * 50)
        logger.info("Hyperparameter Tuning with Randomized Search")
        logger.info("=" * 50)
        
        try:
            # Initialize base model for tuning
            base_estimator = LGBMClassifier(random_state=RANDOM_STATE)
            
            # Setup RandomizedSearchCV
            logger.info(f"Setting up RandomizedSearchCV with {N_ITER_RANDOM_SEARCH} iterations...")
            random_search = RandomizedSearchCV(
                estimator=base_estimator,
                param_distributions=LGBM_PARAM_GRID,
                cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
                scoring=SCORING_METRIC,
                n_iter=N_ITER_RANDOM_SEARCH,
                n_jobs=-1,
                random_state=RANDOM_STATE,
                verbose=1
            )
            
            # Fit the search
            logger.info("Starting hyperparameter search...")
            random_search.fit(X_train, y_train)
            
            # Get best model
            self.tuned_model = random_search.best_estimator_
            self.best_params = random_search.best_params_
            
            logger.info(f"\nBest CV Score ({SCORING_METRIC}): {random_search.best_score_:.4f}")
            logger.info(f"\nBest Parameters:")
            for param, value in self.best_params.items():
                logger.info(f"  {param}: {value}")
            
            # Save the tuned model
            save_model(self.tuned_model, TUNED_MODEL_PATH)
            
            logger.info("\nHyperparameter tuning completed successfully")
            
            return self.tuned_model
            
        except Exception as e:
            logger.error(f"Error during hyperparameter tuning: {str(e)}")
            raise
    
    def compare_models(self, base_metrics, tuned_metrics):
        """
        Compare base and tuned model performance
        
        Args:
            base_metrics (dict): Metrics from base model
            tuned_metrics (dict): Metrics from tuned model
        """
        logger.info("\n" + "=" * 50)
        logger.info("Model Comparison Summary")
        logger.info("=" * 50)
        
        base_recall = base_metrics['test_report']['1']['recall']
        tuned_recall = tuned_metrics['test_report']['1']['recall']
        
        base_precision = base_metrics['test_report']['1']['precision']
        tuned_precision = tuned_metrics['test_report']['1']['precision']
        
        base_f1 = base_metrics['test_report']['1']['f1-score']
        tuned_f1 = tuned_metrics['test_report']['1']['f1-score']
        
        logger.info("\nClass 1 (Machine Failure) Metrics:")
        logger.info(f"\n{'Metric':<15} {'Base Model':<15} {'Tuned Model':<15} {'Change':<15}")
        logger.info("-" * 60)
        logger.info(f"{'Recall':<15} {base_recall:<15.4f} {tuned_recall:<15.4f} {tuned_recall-base_recall:+.4f}")
        logger.info(f"{'Precision':<15} {base_precision:<15.4f} {tuned_precision:<15.4f} {tuned_precision-base_precision:+.4f}")
        logger.info(f"{'F1-Score':<15} {base_f1:<15.4f} {tuned_f1:<15.4f} {tuned_f1-base_f1:+.4f}")
        
        # Determine which model is better
        if tuned_recall > base_recall and tuned_precision >= base_precision * 0.8:
            logger.info("\n✅ Tuned model shows overall improvement")
        elif tuned_recall > base_recall:
            logger.info("\n⚠️  Tuned model has better recall but lower precision")
        else:
            logger.info("\n💡 Base model remains competitive")

def run_training(X_train, y_train, X_test, y_test):
    """
    Execute the full training pipeline
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
    
    Returns:
        tuple: (base_model, tuned_model, base_metrics, tuned_metrics)
    """
    trainer = ModelTrainer()
    
    # Train base model
    base_model = trainer.train_base_model(X_train, y_train)
    base_metrics = trainer.evaluate_model(
        base_model, X_train, y_train, X_test, y_test, 
        model_name="Base LightGBM Model"
    )
    
    # Tune model
    tuned_model = trainer.tune_model(X_train, y_train)
    tuned_metrics = trainer.evaluate_model(
        tuned_model, X_train, y_train, X_test, y_test,
        model_name="Tuned LightGBM Model"
    )
    
    # Compare models
    trainer.compare_models(base_metrics, tuned_metrics)
    
    return base_model, tuned_model, base_metrics, tuned_metrics