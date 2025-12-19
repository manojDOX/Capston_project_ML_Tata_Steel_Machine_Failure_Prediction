"""
Model testing module for TATA Steel Machine Failure Prediction
Provides detailed evaluation and analysis of trained models
"""
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from src.logger import setup_logger
from src.utils import load_model
from src.config import BASE_MODEL_PATH, TUNED_MODEL_PATH, LOG_DIR
import os

logger = setup_logger('testing')

class ModelTester:
    """
    Class to handle comprehensive model testing and evaluation
    """
    
    def __init__(self, model=None, model_path=None):
        """
        Initialize tester with a model or load from path
        
        Args:
            model: Trained model object
            model_path (str): Path to saved model file
        """
        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = load_model(model_path)
        else:
            raise ValueError("Either model or model_path must be provided")
    
    def generate_detailed_report(self, X_test, y_test):
        """
        Generate comprehensive evaluation report
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
        
        Returns:
            dict: Dictionary containing all evaluation metrics
        """
        logger.info("Generating detailed evaluation report...")
        
        try:
            # Get predictions
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            
            # Classification report
            clf_report = classification_report(y_test, y_pred, output_dict=True)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # ROC-AUC
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            # Precision-Recall AUC
            pr_auc = average_precision_score(y_test, y_pred_proba)
            
            # Compile results
            results = {
                'classification_report': clf_report,
                'confusion_matrix': cm,
                'roc_auc_score': roc_auc,
                'precision_recall_auc': pr_auc,
                'predictions': y_pred,
                'prediction_probabilities': y_pred_proba
            }
            
            # Log key metrics
            logger.info("\n--- Test Set Performance Summary ---")
            logger.info(f"ROC-AUC Score: {roc_auc:.4f}")
            logger.info(f"Precision-Recall AUC: {pr_auc:.4f}")
            logger.info(f"\nClass 1 (Machine Failure) Metrics:")
            logger.info(f"  Recall: {clf_report['1']['recall']:.4f}")
            logger.info(f"  Precision: {clf_report['1']['precision']:.4f}")
            logger.info(f"  F1-Score: {clf_report['1']['f1-score']:.4f}")
            logger.info(f"\nConfusion Matrix:")
            logger.info(f"  TN: {cm[0,0]}, FP: {cm[0,1]}")
            logger.info(f"  FN: {cm[1,0]}, TP: {cm[1,1]}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error generating evaluation report: {str(e)}")
            raise
    
    def analyze_feature_importance(self, feature_names):
        """
        Analyze and log feature importance
        
        Args:
            feature_names (list): List of feature names
        
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        logger.info("\nAnalyzing feature importance...")
        
        try:
            # Get feature importance
            importance = self.model.feature_importances_
            
            # Create dataframe
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            # Log top features
            logger.info("\nTop 10 Most Important Features:")
            for idx, row in feature_importance_df.head(10).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
            
            return feature_importance_df
            
        except Exception as e:
            logger.error(f"Error analyzing feature importance: {str(e)}")
            raise
    
    def plot_confusion_matrix(self, cm, save_path=None):
        """
        Plot confusion matrix
        
        Args:
            cm (np.array): Confusion matrix
            save_path (str): Path to save the plot
        """
        try:
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['No Failure', 'Failure'],
                       yticklabels=['No Failure', 'Failure'])
            plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Confusion matrix plot saved to {save_path}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {str(e)}")
    
    def plot_roc_curve(self, y_test, y_pred_proba, save_path=None):
        """
        Plot ROC curve
        
        Args:
            y_test: True labels
            y_pred_proba: Prediction probabilities
            save_path (str): Path to save the plot
        """
        try:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                    label='Random Classifier')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve', 
                     fontsize=14, fontweight='bold')
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"ROC curve plot saved to {save_path}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting ROC curve: {str(e)}")
    
    def plot_precision_recall_curve(self, y_test, y_pred_proba, save_path=None):
        """
        Plot Precision-Recall curve
        
        Args:
            y_test: True labels
            y_pred_proba: Prediction probabilities
            save_path (str): Path to save the plot
        """
        try:
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = average_precision_score(y_test, y_pred_proba)
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='blue', lw=2,
                    label=f'PR curve (AUC = {pr_auc:.3f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
            plt.legend(loc="lower left")
            plt.grid(alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Precision-Recall curve plot saved to {save_path}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting Precision-Recall curve: {str(e)}")

def run_testing(X_test, y_test, model_path=TUNED_MODEL_PATH, generate_plots=True):
    """
    Execute comprehensive model testing
    
    Args:
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        model_path (str): Path to the model to test
        generate_plots (bool): Whether to generate visualization plots
    
    Returns:
        dict: Evaluation results
    """
    logger.info("=" * 50)
    logger.info("Starting Model Testing")
    logger.info("=" * 50)
    
    # Initialize tester
    tester = ModelTester(model_path=model_path)
    
    # Generate detailed report
    results = tester.generate_detailed_report(X_test, y_test)
    
    # Analyze feature importance
    feature_importance = tester.analyze_feature_importance(X_test.columns.tolist())
    results['feature_importance'] = feature_importance
    
    # Generate plots if requested
    if generate_plots:
        plots_dir = os.path.join(LOG_DIR, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Confusion matrix
        cm_path = os.path.join(plots_dir, 'confusion_matrix.png')
        tester.plot_confusion_matrix(results['confusion_matrix'], cm_path)
        
        # ROC curve
        roc_path = os.path.join(plots_dir, 'roc_curve.png')
        tester.plot_roc_curve(y_test, results['prediction_probabilities'], roc_path)
        
        # Precision-Recall curve
        pr_path = os.path.join(plots_dir, 'precision_recall_curve.png')
        tester.plot_precision_recall_curve(y_test, results['prediction_probabilities'], pr_path)
    
    logger.info("\nModel testing completed successfully")
    
    return results