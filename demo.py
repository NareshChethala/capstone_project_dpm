#!/usr/bin/env python3
"""
Credit Risk Prediction Model - Simple Demo

This script demonstrates a simplified version of the credit risk modeling workflow.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from credit_risk_model import DataPreprocessor, ModelTrainer, ModelEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """
    Main function to run the simplified credit risk prediction workflow
    """
    logger.info("Starting Credit Risk Prediction Model Demo")
    
    try:
        # Initialize components
        data_preprocessor = DataPreprocessor(random_state=42)
        model_trainer = ModelTrainer(random_state=42)
        model_evaluator = ModelEvaluator()
        
        # Step 1: Create Sample Data
        logger.info("Step 1: Creating Sample Dataset")
        X, y = data_preprocessor.create_sample_dataset(n_samples=2000, n_features=10)
        
        logger.info(f"Dataset shape: {X.shape}")
        logger.info(f"Target distribution:\n{y.value_counts()}")
        logger.info(f"Features: {list(X.columns)}")
        
        # Step 2: Data Preprocessing
        logger.info("Step 2: Data Preprocessing")
        X_train, X_test, y_train, y_test = data_preprocessor.preprocess_pipeline(
            X, y, test_size=0.2, handle_outliers=True
        )
        
        # Step 3: Model Training
        logger.info("Step 3: Training Multiple Models")
        
        # Train a subset of models for demonstration
        models_to_train = ['logistic_regression', 'random_forest', 'xgboost']
        
        trained_models = model_trainer.train_all_models(
            X_train, y_train,
            models_to_train=models_to_train,
            use_grid_search=False,  # Set to False for faster demo
            handle_imbalance='smote'
        )
        
        # Display training summary
        summary = model_trainer.get_model_summary()
        logger.info("Training Summary:")
        logger.info(f"\n{summary}")
        
        # Step 4: Model Evaluation
        logger.info("Step 4: Model Evaluation")
        
        # Compare all models
        comparison_results = model_evaluator.compare_models(
            trained_models, X_test, y_test
        )
        
        logger.info("Model Comparison Results:")
        logger.info(f"\n{comparison_results}")
        
        # Step 5: Generate Outputs
        logger.info("Step 5: Generating Outputs")
        
        # Create output directory
        os.makedirs('outputs', exist_ok=True)
        
        # Save comparison results
        comparison_results.to_csv('outputs/model_comparison_results.csv', index=False)
        
        # Get best model
        best_model_name = comparison_results.iloc[0]['Model']
        best_model = trained_models[best_model_name]
        
        # Generate detailed evaluation
        detailed_evaluation = model_evaluator.evaluate_single_model(
            best_model, X_test, y_test, best_model_name
        )
        
        # Generate and save report
        report = model_evaluator.generate_evaluation_report(best_model_name)
        logger.info(report)
        
        with open('outputs/model_evaluation_report.txt', 'w') as f:
            f.write(report)
        
        # Create visualizations
        plt.style.use('default')
        
        # Model comparison visualization
        fig = model_evaluator.plot_model_comparison(trained_models, X_test, y_test)
        plt.savefig('outputs/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Individual model analysis
        y_prob_best = best_model.predict_proba(X_test)[:, 1]
        y_pred_best = best_model.predict(X_test)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # ROC Curve
        model_evaluator.plot_roc_curve(y_test, y_prob_best, best_model_name, axes[0, 0])
        
        # Precision-Recall Curve
        model_evaluator.plot_precision_recall_curve(y_test, y_prob_best, best_model_name, axes[0, 1])
        
        # Confusion Matrix
        model_evaluator.plot_confusion_matrix(y_test, y_pred_best, best_model_name, axes[1, 0])
        
        # Calibration Curve
        model_evaluator.plot_calibration_curve(y_test, y_prob_best, best_model_name, 10, axes[1, 1])
        
        plt.tight_layout()
        plt.savefig('outputs/best_model_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Feature importance if available
        if hasattr(best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': X_test.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 8))
            top_features = importance_df.head(10)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'Top 10 Feature Importances - {best_model_name}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('outputs/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save feature importance to CSV
            importance_df.to_csv('outputs/feature_importance.csv', index=False)
        
        # Save sample predictions
        predictions_df = pd.DataFrame({
            'true_label': y_test,
            'predicted_label': y_pred_best,
            'predicted_probability': y_prob_best
        })
        predictions_df.to_csv('outputs/sample_predictions.csv', index=False)
        
        # Save models
        os.makedirs('models', exist_ok=True)
        model_trainer.save_models(trained_models, 'models/credit_risk_model')
        
        logger.info("Credit Risk Prediction Demo Completed Successfully!")
        logger.info("Results saved in 'outputs/' directory")
        logger.info("Models saved in 'models/' directory")
        
        # Print final summary
        print("\n" + "="*60)
        print("CREDIT RISK PREDICTION MODEL - DEMO SUMMARY")
        print("="*60)
        print(f"Dataset Size: {len(X)} samples, {X.shape[1]} features")
        print(f"Best Model: {best_model_name}")
        print(f"ROC-AUC Score: {comparison_results.iloc[0]['ROC_AUC']:.4f}")
        print(f"Precision: {comparison_results.iloc[0]['Precision']:.4f}")
        print(f"Recall: {comparison_results.iloc[0]['Recall']:.4f}")
        print(f"F1-Score: {comparison_results.iloc[0]['F1_Score']:.4f}")
        print(f"Net Profit: ${comparison_results.iloc[0]['Net_Profit']:,.2f}")
        print("\nFiles Generated:")
        print("- outputs/model_comparison_results.csv")
        print("- outputs/model_evaluation_report.txt")
        print("- outputs/model_comparison.png")
        print("- outputs/best_model_analysis.png")
        print("- outputs/feature_importance.png")
        print("- outputs/sample_predictions.csv")
        print("- models/credit_risk_model_*.joblib")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error in demo workflow: {str(e)}")
        raise

if __name__ == "__main__":
    main()