"""
Basic tests for the credit risk model components
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from credit_risk_model import DataPreprocessor, FeatureEngineer, ModelTrainer, ModelEvaluator

class TestDataPreprocessor:
    def test_create_sample_dataset(self):
        preprocessor = DataPreprocessor()
        X, y = preprocessor.create_sample_dataset(n_samples=100, n_features=5)
        
        assert X.shape == (100, 5)
        assert len(y) == 100
        assert y.dtype == int
        assert set(y.unique()).issubset({0, 1})
    
    def test_handle_missing_values(self):
        preprocessor = DataPreprocessor()
        
        # Create data with missing values
        data = pd.DataFrame({
            'A': [1, 2, np.nan, 4],
            'B': [1, np.nan, 3, 4],
            'C': ['a', 'b', np.nan, 'd']
        })
        
        result = preprocessor.handle_missing_values(data, strategy='median')
        assert result.isnull().sum().sum() == 0

class TestFeatureEngineer:
    def test_calculate_financial_ratios(self):
        engineer = FeatureEngineer()
        
        data = pd.DataFrame({
            'total_assets': [100, 200, 300],
            'total_liabilities': [50, 100, 150],
            'revenue': [20, 40, 60],
            'net_income': [5, 10, 15]
        })
        
        result = engineer.calculate_financial_ratios(data)
        assert 'return_on_assets' in result.columns
        assert len(result.columns) > len(data.columns)
    
    def test_create_interaction_features(self):
        engineer = FeatureEngineer()
        
        data = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': [7, 8, 9]
        })
        
        result = engineer.create_interaction_features(data, max_interactions=2)
        assert len(result.columns) > len(data.columns)

class TestModelTrainer:
    def test_initialize_models(self):
        trainer = ModelTrainer()
        models = trainer.initialize_models()
        
        assert 'logistic_regression' in models
        assert 'random_forest' in models
        assert 'xgboost' in models
        assert len(models) > 0
    
    def test_handle_class_imbalance(self):
        trainer = ModelTrainer()
        
        X = pd.DataFrame({'A': range(100), 'B': range(100, 200)})
        y = pd.Series([0] * 90 + [1] * 10)  # Imbalanced
        
        X_balanced, y_balanced = trainer.handle_class_imbalance(X, y, method='smote')
        
        # Check that classes are more balanced
        original_ratio = sum(y) / len(y)
        new_ratio = sum(y_balanced) / len(y_balanced)
        
        assert abs(new_ratio - 0.5) < abs(original_ratio - 0.5)

class TestModelEvaluator:
    def test_calculate_classification_metrics(self):
        evaluator = ModelEvaluator()
        
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        y_prob = np.array([0.1, 0.4, 0.6, 0.9])
        
        metrics = evaluator.calculate_classification_metrics(y_true, y_pred, y_prob)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'roc_auc' in metrics
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_calculate_business_metrics(self):
        evaluator = ModelEvaluator()
        
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        y_prob = np.array([0.1, 0.4, 0.6, 0.9])
        
        business_metrics = evaluator.calculate_business_metrics(y_true, y_pred, y_prob)
        
        assert 'net_profit' in business_metrics
        assert 'total_revenue' in business_metrics
        assert 'approval_rate' in business_metrics

if __name__ == "__main__":
    pytest.main([__file__])