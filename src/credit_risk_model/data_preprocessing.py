"""
Data Preprocessing Module for Credit Risk Modeling

This module handles data loading, cleaning, and initial preprocessing for credit risk datasets.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging

class DataPreprocessor:
    """
    Handles data preprocessing tasks for credit risk modeling including:
    - Data loading and validation
    - Missing value treatment
    - Outlier detection and handling
    - Data type conversions
    - Train/test split
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.logger = logging.getLogger(__name__)
        
    def load_data(self, file_path, target_column='default'):
        """
        Load dataset from file
        
        Args:
            file_path (str): Path to the dataset file
            target_column (str): Name of the target variable column
            
        Returns:
            tuple: (X, y) features and target
        """
        try:
            # Support multiple file formats
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format. Use .csv or .xlsx")
                
            self.logger.info(f"Loaded dataset with shape: {df.shape}")
            
            # Separate features and target
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in dataset")
                
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def handle_missing_values(self, X, strategy='median'):
        """
        Handle missing values in the dataset
        
        Args:
            X (pd.DataFrame): Feature dataset
            strategy (str): Strategy for handling missing values
                          ('median', 'mean', 'mode', 'drop')
                          
        Returns:
            pd.DataFrame: Dataset with missing values handled
        """
        X_clean = X.copy()
        
        # Log missing value information
        missing_info = X_clean.isnull().sum()
        if missing_info.sum() > 0:
            self.logger.info(f"Missing values found:\n{missing_info[missing_info > 0]}")
        
        if strategy == 'drop':
            X_clean = X_clean.dropna()
        elif strategy == 'median':
            numeric_cols = X_clean.select_dtypes(include=[np.number]).columns
            X_clean[numeric_cols] = X_clean[numeric_cols].fillna(X_clean[numeric_cols].median())
            
            # For categorical columns, use mode
            categorical_cols = X_clean.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                X_clean[col] = X_clean[col].fillna(X_clean[col].mode()[0] if not X_clean[col].mode().empty else 'Unknown')
                
        elif strategy == 'mean':
            numeric_cols = X_clean.select_dtypes(include=[np.number]).columns
            X_clean[numeric_cols] = X_clean[numeric_cols].fillna(X_clean[numeric_cols].mean())
            
        return X_clean
    
    def detect_outliers(self, X, method='iqr', threshold=1.5):
        """
        Detect outliers in numerical columns
        
        Args:
            X (pd.DataFrame): Feature dataset
            method (str): Method for outlier detection ('iqr', 'zscore')
            threshold (float): Threshold for outlier detection
            
        Returns:
            pd.DataFrame: Boolean mask indicating outliers
        """
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        outlier_mask = pd.DataFrame(False, index=X.index, columns=X.columns)
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outlier_mask[col] = (X[col] < lower_bound) | (X[col] > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs((X[col] - X[col].mean()) / X[col].std())
                outlier_mask[col] = z_scores > threshold
                
        return outlier_mask
    
    def encode_categorical_variables(self, X):
        """
        Encode categorical variables using label encoding
        
        Args:
            X (pd.DataFrame): Feature dataset
            
        Returns:
            pd.DataFrame: Dataset with encoded categorical variables
        """
        X_encoded = X.copy()
        categorical_cols = X_encoded.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X_encoded[col] = self.label_encoders[col].fit_transform(X_encoded[col].astype(str))
            else:
                X_encoded[col] = self.label_encoders[col].transform(X_encoded[col].astype(str))
                
        return X_encoded
    
    def scale_features(self, X_train, X_test=None):
        """
        Scale numerical features using StandardScaler
        
        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame, optional): Test features
            
        Returns:
            tuple: Scaled training and test sets (if provided)
        """
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        
        # Fit scaler on training data
        X_train_scaled = X_train.copy()
        X_train_scaled[numeric_cols] = self.scaler.fit_transform(X_train[numeric_cols])
        
        if X_test is not None:
            X_test_scaled = X_test.copy()
            X_test_scaled[numeric_cols] = self.scaler.transform(X_test[numeric_cols])
            return X_train_scaled, X_test_scaled
            
        return X_train_scaled
    
    def preprocess_pipeline(self, X, y, test_size=0.2, handle_outliers=True):
        """
        Complete preprocessing pipeline
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
            test_size (float): Proportion of data for testing
            handle_outliers (bool): Whether to detect and log outliers
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test) preprocessed and split data
        """
        self.logger.info("Starting preprocessing pipeline...")
        
        # Handle missing values
        X_clean = self.handle_missing_values(X)
        
        # Detect outliers (for logging purposes)
        if handle_outliers:
            outlier_mask = self.detect_outliers(X_clean)
            outlier_count = outlier_mask.any(axis=1).sum()
            self.logger.info(f"Detected {outlier_count} rows with outliers")
        
        # Encode categorical variables
        X_encoded = self.encode_categorical_variables(X_clean)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=test_size, 
            random_state=self.random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        self.logger.info(f"Preprocessing completed. Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def create_sample_dataset(self, n_samples=1000, n_features=10):
        """
        Create a sample credit risk dataset for demonstration
        
        Args:
            n_samples (int): Number of samples
            n_features (int): Number of features
            
        Returns:
            tuple: (X, y) sample features and target
        """
        np.random.seed(self.random_state)
        
        # Create feature names relevant to credit risk
        feature_names = [
            'total_assets', 'total_liabilities', 'revenue', 'net_income',
            'cash_flow', 'debt_to_equity', 'current_ratio', 'quick_ratio',
            'return_on_assets', 'return_on_equity', 'profit_margin',
            'asset_turnover', 'leverage_ratio', 'interest_coverage',
            'working_capital', 'company_age', 'industry_risk_score',
            'market_cap', 'employee_count', 'credit_rating_score'
        ]
        
        selected_features = feature_names[:n_features]
        
        # Generate correlated features that make sense for credit risk
        X = pd.DataFrame()
        
        # Financial metrics (in millions)
        X['total_assets'] = np.random.lognormal(mean=5, sigma=1.5, size=n_samples)
        X['total_liabilities'] = X['total_assets'] * np.random.uniform(0.3, 0.8, n_samples)
        X['revenue'] = X['total_assets'] * np.random.uniform(0.1, 0.5, n_samples)
        X['net_income'] = X['revenue'] * np.random.uniform(-0.1, 0.15, n_samples)
        
        # Financial ratios
        X['debt_to_equity'] = X['total_liabilities'] / (X['total_assets'] - X['total_liabilities'])
        X['current_ratio'] = np.random.uniform(0.5, 3.0, n_samples)
        X['quick_ratio'] = X['current_ratio'] * np.random.uniform(0.7, 1.0, n_samples)
        X['return_on_assets'] = X['net_income'] / X['total_assets']
        X['return_on_equity'] = X['net_income'] / (X['total_assets'] - X['total_liabilities'])
        X['profit_margin'] = X['net_income'] / X['revenue']
        
        # Additional features
        if n_features > 10:
            X['asset_turnover'] = X['revenue'] / X['total_assets']
            X['leverage_ratio'] = X['total_liabilities'] / X['total_assets']
            X['interest_coverage'] = np.random.uniform(-5, 20, n_samples)
            X['working_capital'] = np.random.normal(0, X['total_assets'].std(), n_samples)
            X['company_age'] = np.random.randint(1, 50, n_samples)
            X['industry_risk_score'] = np.random.uniform(1, 10, n_samples)
            X['market_cap'] = X['total_assets'] * np.random.uniform(0.5, 2.0, n_samples)
            X['employee_count'] = np.random.lognormal(mean=4, sigma=1, size=n_samples)
            X['credit_rating_score'] = np.random.uniform(300, 850, n_samples)
        
        # Select only the requested features that exist
        available_features = [f for f in selected_features if f in X.columns]
        if len(available_features) < n_features:
            # If we don't have enough features, take the first n_features available
            available_features = X.columns[:n_features].tolist()
        X = X[available_features]
        
        # Create target variable (default probability based on financial health)
        # Use only the features that actually exist in the dataframe
        financial_health_score = (
            -X.get('debt_to_equity', 0) * 0.3 +
            X.get('current_ratio', 1.5) * 0.2 +
            X.get('return_on_assets', 0.05) * 0.3 +
            X.get('profit_margin', 0.05) * 0.2
        )
        
        # Convert to binary default indicator (1 = default, 0 = no default)
        default_probability = 1 / (1 + np.exp(financial_health_score))  # Sigmoid
        y = (np.random.random(n_samples) < default_probability).astype(int)
        
        self.logger.info(f"Created sample dataset with {n_samples} samples and {n_features} features")
        self.logger.info(f"Default rate: {y.mean():.2%}")
        
        return X, y