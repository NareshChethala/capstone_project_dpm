"""
Model Training Module for Credit Risk Modeling

This module handles training of multiple machine learning models for credit risk prediction,
including ensemble methods and hyperparameter optimization.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import xgboost as xgb
import lightgbm as lgb
import joblib
import logging
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """
    Handles training multiple machine learning models for credit risk prediction including:
    - Traditional ML algorithms (Logistic Regression, Random Forest, SVM)
    - Gradient boosting models (XGBoost, LightGBM, CatBoost)
    - Hyperparameter optimization
    - Model comparison and selection
    - Handling class imbalance
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.best_models = {}
        self.cv_scores = {}
        self.logger = logging.getLogger(__name__)
        
    def handle_class_imbalance(self, X, y, method='smote'):
        """
        Handle class imbalance in the dataset
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
            method (str): Resampling method ('smote', 'undersample', 'smoteenn', 'none')
            
        Returns:
            tuple: Resampled X and y
        """
        if method == 'none':
            return X, y
            
        class_counts = y.value_counts()
        self.logger.info(f"Original class distribution: {class_counts.to_dict()}")
        
        if method == 'smote':
            resampler = SMOTE(random_state=self.random_state)
        elif method == 'undersample':
            resampler = RandomUnderSampler(random_state=self.random_state)
        elif method == 'smoteenn':
            resampler = SMOTEENN(random_state=self.random_state)
        else:
            raise ValueError("Method must be one of: 'smote', 'undersample', 'smoteenn', 'none'")
            
        X_resampled, y_resampled = resampler.fit_resample(X, y)
        
        new_class_counts = pd.Series(y_resampled).value_counts()
        self.logger.info(f"Resampled class distribution: {new_class_counts.to_dict()}")
        
        return X_resampled, y_resampled
    
    def initialize_models(self):
        """
        Initialize all models with default parameters
        
        Returns:
            dict: Dictionary of initialized models
        """
        models = {
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000
            ),
            'random_forest': RandomForestClassifier(
                random_state=self.random_state,
                n_estimators=100
            ),
            'gradient_boosting': GradientBoostingClassifier(
                random_state=self.random_state,
                n_estimators=100
            ),
            'xgboost': xgb.XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss'
            ),
            'lightgbm': lgb.LGBMClassifier(
                random_state=self.random_state,
                verbose=-1
            ),
            'svm': SVC(
                random_state=self.random_state,
                probability=True
            )
        }
        
        self.models = models
        return models
    
    def get_hyperparameter_grids(self):
        """
        Define hyperparameter grids for GridSearchCV
        
        Returns:
            dict: Dictionary of parameter grids for each model
        """
        param_grids = {
            'logistic_regression': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'lightgbm': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'svm': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto', 0.001, 0.01]
            }
        }
        
        return param_grids
    
    def train_single_model(self, model_name, X_train, y_train, use_grid_search=True):
        """
        Train a single model with optional hyperparameter optimization
        
        Args:
            model_name (str): Name of the model to train
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            use_grid_search (bool): Whether to use GridSearchCV for hyperparameter tuning
            
        Returns:
            sklearn estimator: Trained model
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        model = self.models[model_name]
        
        if use_grid_search:
            param_grids = self.get_hyperparameter_grids()
            
            if model_name in param_grids:
                self.logger.info(f"Starting hyperparameter optimization for {model_name}")
                
                # Use smaller parameter grid for faster training if dataset is large
                if X_train.shape[0] > 10000:
                    # Reduce parameter grid for large datasets
                    param_grid = {k: v[:2] if isinstance(v, list) and len(v) > 2 else v 
                                for k, v in param_grids[model_name].items()}
                else:
                    param_grid = param_grids[model_name]
                
                # Cross-validation setup
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
                
                grid_search = GridSearchCV(
                    model, 
                    param_grid, 
                    cv=cv,
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                
                self.logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
                self.logger.info(f"Best CV score for {model_name}: {grid_search.best_score_:.4f}")
                
                self.best_models[model_name] = best_model
                return best_model
        
        # Train with default parameters
        self.logger.info(f"Training {model_name} with default parameters")
        model.fit(X_train, y_train)
        return model
    
    def train_all_models(self, X_train, y_train, models_to_train=None, 
                        use_grid_search=True, handle_imbalance='smote'):
        """
        Train multiple models and compare their performance
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            models_to_train (list, optional): List of model names to train
            use_grid_search (bool): Whether to use hyperparameter optimization
            handle_imbalance (str): Method to handle class imbalance
            
        Returns:
            dict: Dictionary of trained models
        """
        if not self.models:
            self.initialize_models()
        
        if models_to_train is None:
            models_to_train = list(self.models.keys())
        
        # Handle class imbalance
        X_train_balanced, y_train_balanced = self.handle_class_imbalance(
            X_train, y_train, method=handle_imbalance
        )
        
        trained_models = {}
        
        for model_name in models_to_train:
            try:
                self.logger.info(f"Training {model_name}...")
                
                trained_model = self.train_single_model(
                    model_name, X_train_balanced, y_train_balanced, use_grid_search
                )
                
                trained_models[model_name] = trained_model
                
                # Perform cross-validation on original (unbalanced) data
                cv_scores = cross_val_score(
                    trained_model, X_train, y_train, 
                    cv=3, scoring='roc_auc'
                )
                
                self.cv_scores[model_name] = {
                    'mean': cv_scores.mean(),
                    'std': cv_scores.std(),
                    'scores': cv_scores
                }
                
                self.logger.info(f"{model_name} CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
            except Exception as e:
                self.logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        self.logger.info(f"Training completed for {len(trained_models)} models")
        return trained_models
    
    def ensemble_predictions(self, models, X, method='voting', weights=None):
        """
        Create ensemble predictions from multiple models
        
        Args:
            models (dict): Dictionary of trained models
            X (pd.DataFrame): Features for prediction
            method (str): Ensemble method ('voting', 'weighted_average')
            weights (list, optional): Weights for weighted average
            
        Returns:
            tuple: (predictions, prediction_probabilities)
        """
        if method == 'voting':
            # Simple majority voting
            predictions = []
            probabilities = []
            
            for model_name, model in models.items():
                pred = model.predict(X)
                prob = model.predict_proba(X)[:, 1]  # Probability of positive class
                predictions.append(pred)
                probabilities.append(prob)
            
            # Majority vote for predictions
            predictions_array = np.array(predictions)
            ensemble_pred = np.apply_along_axis(
                lambda x: np.bincount(x).argmax(), axis=0, arr=predictions_array
            )
            
            # Average probabilities
            probabilities_array = np.array(probabilities)
            ensemble_prob = np.mean(probabilities_array, axis=0)
            
        elif method == 'weighted_average':
            if weights is None:
                # Use CV scores as weights
                weights = [self.cv_scores.get(name, {}).get('mean', 1.0) 
                          for name in models.keys()]
            
            if len(weights) != len(models):
                raise ValueError("Number of weights must match number of models")
            
            # Normalize weights
            weights = np.array(weights) / np.sum(weights)
            
            probabilities = []
            for model_name, model in models.items():
                prob = model.predict_proba(X)[:, 1]
                probabilities.append(prob)
            
            # Weighted average of probabilities
            probabilities_array = np.array(probabilities)
            ensemble_prob = np.average(probabilities_array, axis=0, weights=weights)
            ensemble_pred = (ensemble_prob > 0.5).astype(int)
        
        return ensemble_pred, ensemble_prob
    
    def get_feature_importance(self, model, feature_names):
        """
        Extract feature importance from trained model
        
        Args:
            model: Trained model
            feature_names (list): List of feature names
            
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        importance_dict = {}
        
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importance_dict['importance'] = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Linear models
            importance_dict['importance'] = np.abs(model.coef_[0])
        else:
            # For models without built-in feature importance
            return None
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_dict['importance']
        })
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        return importance_df
    
    def save_models(self, models, filepath_prefix):
        """
        Save trained models to disk
        
        Args:
            models (dict): Dictionary of trained models
            filepath_prefix (str): Prefix for model file paths
        """
        for model_name, model in models.items():
            filepath = f"{filepath_prefix}_{model_name}.joblib"
            joblib.dump(model, filepath)
            self.logger.info(f"Saved {model_name} to {filepath}")
    
    def load_models(self, filepaths):
        """
        Load trained models from disk
        
        Args:
            filepaths (dict): Dictionary mapping model names to file paths
            
        Returns:
            dict: Dictionary of loaded models
        """
        loaded_models = {}
        
        for model_name, filepath in filepaths.items():
            try:
                model = joblib.load(filepath)
                loaded_models[model_name] = model
                self.logger.info(f"Loaded {model_name} from {filepath}")
            except Exception as e:
                self.logger.error(f"Error loading {model_name}: {str(e)}")
        
        return loaded_models
    
    def get_model_summary(self):
        """
        Get summary of all trained models and their performance
        
        Returns:
            pd.DataFrame: Summary of model performance
        """
        if not self.cv_scores:
            self.logger.warning("No models have been trained yet")
            return None
        
        summary_data = []
        
        for model_name, scores in self.cv_scores.items():
            summary_data.append({
                'Model': model_name,
                'CV_AUC_Mean': scores['mean'],
                'CV_AUC_Std': scores['std'],
                'CV_AUC_Min': scores['scores'].min(),
                'CV_AUC_Max': scores['scores'].max()
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('CV_AUC_Mean', ascending=False)
        
        return summary_df