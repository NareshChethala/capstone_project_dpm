#!/usr/bin/env python3
"""
Credit Risk Prediction Model - Main Application

This script demonstrates the complete workflow for building and evaluating
credit risk prediction models using multiple variables and machine learning techniques.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from credit_risk_model import DataPreprocessor, FeatureEngineer, ModelTrainer, ModelEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('credit_risk_model.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def main():
    """
    Main function to run the credit risk prediction workflow
    """
    logger.info("Starting Credit Risk Prediction Model Workflow")
    
    try:
        # Initialize components
        data_preprocessor = DataPreprocessor(random_state=42)
        feature_engineer = FeatureEngineer()
        model_trainer = ModelTrainer(random_state=42)
        model_evaluator = ModelEvaluator()
        
        # Step 1: Create or Load Data
        logger.info("Step 1: Loading/Creating Dataset")
        
        # For demonstration, create a sample dataset
        # In practice, replace this with: X, y = data_preprocessor.load_data('your_data.csv')
        X, y = data_preprocessor.create_sample_dataset(n_samples=5000, n_features=15)
        
        logger.info(f"Dataset shape: {X.shape}")
        logger.info(f"Target distribution:\n{y.value_counts()}")
        
        # Step 2: Data Preprocessing
        logger.info("Step 2: Data Preprocessing")
        
        X_train, X_test, y_train, y_test = data_preprocessor.preprocess_pipeline(
            X, y, test_size=0.2, handle_outliers=True
        )
        
        # Step 3: Feature Engineering
        logger.info("Step 3: Feature Engineering")
        
        X_train_engineered = feature_engineer.feature_engineering_pipeline(
            X_train, y_train,
            include_interactions=True,
            include_polynomials=False,  # Skip for demo to reduce complexity
            include_pca=False,
            feature_selection_k=20
        )
        
        X_test_engineered = feature_engineer.feature_engineering_pipeline(
            X_test, y_test,
            include_interactions=True,
            include_polynomials=False,
            include_pca=False,
            feature_selection_k=20
        )
        
        logger.info(f"Engineered features shape: {X_train_engineered.shape}")
        
        # Step 4: Model Training
        logger.info("Step 4: Training Multiple Models")
        
        # Train a subset of models for demonstration
        models_to_train = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm']
        
        trained_models = model_trainer.train_all_models(
            X_train_engineered, y_train,
            models_to_train=models_to_train,
            use_grid_search=False,  # Set to False for faster demo
            handle_imbalance='smote'
        )
        
        # Display training summary
        summary = model_trainer.get_model_summary()
        logger.info("Training Summary:")
        logger.info(f"\n{summary}")
        
        # Step 5: Model Evaluation
        logger.info("Step 5: Model Evaluation")
        
        # Compare all models
        comparison_results = model_evaluator.compare_models(
            trained_models, X_test_engineered, y_test
        )
        
        logger.info("Model Comparison Results:")
        logger.info(f"\n{comparison_results}")
        
        # Generate detailed report for best model
        best_model_name = comparison_results.iloc[0]['Model']
        best_model = trained_models[best_model_name]
        
        # Evaluate best model in detail
        detailed_evaluation = model_evaluator.evaluate_single_model(
            best_model, X_test_engineered, y_test, best_model_name
        )
        
        # Generate and print detailed report
        report = model_evaluator.generate_evaluation_report(best_model_name)
        logger.info(report)
        
        # Step 6: Visualizations
        logger.info("Step 6: Generating Visualizations")
        
        # Create output directory
        os.makedirs('outputs', exist_ok=True)
        
        # Plot model comparison
        plt.style.use('seaborn-v0_8')
        fig = model_evaluator.plot_model_comparison(
            trained_models, X_test_engineered, y_test
        )
        plt.savefig('outputs/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create individual plots for best model
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        y_prob_best = best_model.predict_proba(X_test_engineered)[:, 1]
        y_pred_best = best_model.predict(X_test_engineered)
        
        # ROC Curve
        model_evaluator.plot_roc_curve(y_test, y_prob_best, best_model_name, axes[0, 0])
        
        # Precision-Recall Curve
        model_evaluator.plot_precision_recall_curve(y_test, y_prob_best, best_model_name, axes[0, 1])
        
        # Confusion Matrix
        model_evaluator.plot_confusion_matrix(y_test, y_pred_best, best_model_name, axes[1, 0])
        
        # Calibration Curve
        model_evaluator.plot_calibration_curve(y_test, y_prob_best, best_model_name, axes[1, 1])
        
        plt.tight_layout()
        plt.savefig('outputs/best_model_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Feature importance analysis
        if hasattr(best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': X_test_engineered.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 8))
            top_features = importance_df.head(15)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'Top 15 Feature Importances - {best_model_name}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('outputs/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save feature importance to CSV
            importance_df.to_csv('outputs/feature_importance.csv', index=False)
        
        # Step 7: Save Results
        logger.info("Step 7: Saving Results")
        
        # Save models
        model_trainer.save_models(trained_models, 'models/credit_risk_model')
        
        # Save comparison results
        comparison_results.to_csv('outputs/model_comparison_results.csv', index=False)
        
        # Save sample predictions
        predictions_df = pd.DataFrame({
            'true_label': y_test,
            'predicted_label': y_pred_best,
            'predicted_probability': y_prob_best
        })
        predictions_df.to_csv('outputs/sample_predictions.csv', index=False)
        
        logger.info("Credit Risk Prediction Workflow Completed Successfully!")
        logger.info("Results saved in 'outputs/' directory")
        logger.info("Models saved in 'models/' directory")
        
        # Print final summary
        print("\n" + "="*60)
        print("CREDIT RISK PREDICTION MODEL - FINAL SUMMARY")
        print("="*60)
        print(f"Best Model: {best_model_name}")
        print(f"ROC-AUC Score: {comparison_results.iloc[0]['ROC_AUC']:.4f}")
        print(f"Precision: {comparison_results.iloc[0]['Precision']:.4f}")
        print(f"Recall: {comparison_results.iloc[0]['Recall']:.4f}")
        print(f"F1-Score: {comparison_results.iloc[0]['F1_Score']:.4f}")
        print(f"Net Profit: ${comparison_results.iloc[0]['Net_Profit']:,.2f}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error in main workflow: {str(e)}")
        raise

if __name__ == "__main__":
    main()