"""
Feature Engineering Module for Credit Risk Modeling

This module handles advanced feature engineering techniques specifically designed
for credit risk assessment and default prediction.
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
import logging

class FeatureEngineer:
    """
    Handles feature engineering tasks for credit risk modeling including:
    - Financial ratio calculations
    - Interaction features
    - Feature selection
    - Dimensionality reduction
    - Industry-specific features
    """
    
    def __init__(self):
        self.feature_selector = None
        self.pca = None
        self.poly_features = None
        self.logger = logging.getLogger(__name__)
        
    def calculate_financial_ratios(self, df):
        """
        Calculate important financial ratios for credit risk assessment
        
        Args:
            df (pd.DataFrame): Dataset with financial data
            
        Returns:
            pd.DataFrame: Dataset with additional financial ratios
        """
        df_ratios = df.copy()
        
        # Liquidity ratios
        if 'current_assets' in df.columns and 'current_liabilities' in df.columns:
            df_ratios['current_ratio'] = df['current_assets'] / (df['current_liabilities'] + 1e-8)
            
        if 'cash' in df.columns and 'current_liabilities' in df.columns:
            df_ratios['cash_ratio'] = df['cash'] / (df['current_liabilities'] + 1e-8)
            
        # Leverage ratios
        if 'total_debt' in df.columns and 'total_equity' in df.columns:
            df_ratios['debt_to_equity'] = df['total_debt'] / (df['total_equity'] + 1e-8)
            
        if 'total_debt' in df.columns and 'total_assets' in df.columns:
            df_ratios['debt_to_assets'] = df['total_debt'] / (df['total_assets'] + 1e-8)
            
        # Profitability ratios
        if 'net_income' in df.columns and 'total_assets' in df.columns:
            df_ratios['return_on_assets'] = df['net_income'] / (df['total_assets'] + 1e-8)
            
        if 'net_income' in df.columns and 'total_equity' in df.columns:
            df_ratios['return_on_equity'] = df['net_income'] / (df['total_equity'] + 1e-8)
            
        if 'net_income' in df.columns and 'revenue' in df.columns:
            df_ratios['profit_margin'] = df['net_income'] / (df['revenue'] + 1e-8)
            
        # Efficiency ratios
        if 'revenue' in df.columns and 'total_assets' in df.columns:
            df_ratios['asset_turnover'] = df['revenue'] / (df['total_assets'] + 1e-8)
            
        if 'cost_of_goods_sold' in df.columns and 'inventory' in df.columns:
            df_ratios['inventory_turnover'] = df['cost_of_goods_sold'] / (df['inventory'] + 1e-8)
            
        # Coverage ratios
        if 'ebit' in df.columns and 'interest_expense' in df.columns:
            df_ratios['interest_coverage'] = df['ebit'] / (df['interest_expense'] + 1e-8)
            
        self.logger.info(f"Added {len(df_ratios.columns) - len(df.columns)} financial ratios")
        return df_ratios
    
    def create_interaction_features(self, df, feature_pairs=None, max_interactions=20):
        """
        Create interaction features between important variables
        
        Args:
            df (pd.DataFrame): Input dataset
            feature_pairs (list): Specific feature pairs to interact
            max_interactions (int): Maximum number of interactions to create
            
        Returns:
            pd.DataFrame: Dataset with interaction features
        """
        df_interactions = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if feature_pairs is None:
            # Create interactions between highly correlated features
            corr_matrix = df[numeric_cols].corr().abs()
            
            # Find pairs with moderate correlation (0.3 to 0.8)
            feature_pairs = []
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    corr = corr_matrix.iloc[i, j]
                    if 0.3 <= corr <= 0.8:
                        feature_pairs.append((numeric_cols[i], numeric_cols[j]))
                        
            # Limit to max_interactions
            feature_pairs = feature_pairs[:max_interactions]
        
        # Create interaction features
        for feat1, feat2 in feature_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                # Multiplicative interaction
                df_interactions[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
                
                # Ratio interaction (avoid division by zero)
                df_interactions[f'{feat1}_div_{feat2}'] = df[feat1] / (df[feat2] + 1e-8)
                
        self.logger.info(f"Created {len(feature_pairs) * 2} interaction features")
        return df_interactions
    
    def create_polynomial_features(self, df, degree=2, include_bias=False):
        """
        Create polynomial features for non-linear relationships
        
        Args:
            df (pd.DataFrame): Input dataset
            degree (int): Degree of polynomial features
            include_bias (bool): Whether to include bias term
            
        Returns:
            pd.DataFrame: Dataset with polynomial features
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Limit to most important features to avoid explosion
        if len(numeric_cols) > 10:
            # Select top 10 features by variance
            variances = df[numeric_cols].var().sort_values(ascending=False)
            numeric_cols = variances.head(10).index.tolist()
        
        self.poly_features = PolynomialFeatures(
            degree=degree, 
            include_bias=include_bias,
            interaction_only=False
        )
        
        poly_array = self.poly_features.fit_transform(df[numeric_cols])
        poly_feature_names = self.poly_features.get_feature_names_out(numeric_cols)
        
        # Create new dataframe with polynomial features
        df_poly = pd.DataFrame(poly_array, columns=poly_feature_names, index=df.index)
        
        # Combine with non-numeric columns
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            df_combined = pd.concat([df_poly, df[non_numeric_cols]], axis=1)
        else:
            df_combined = df_poly
            
        self.logger.info(f"Created polynomial features with degree {degree}")
        return df_combined
    
    def create_aggregate_features(self, df):
        """
        Create aggregate features from existing variables
        
        Args:
            df (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Dataset with aggregate features
        """
        df_agg = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 3:
            # Statistical aggregates
            df_agg['feature_mean'] = df[numeric_cols].mean(axis=1)
            df_agg['feature_std'] = df[numeric_cols].std(axis=1)
            df_agg['feature_skew'] = df[numeric_cols].skew(axis=1)
            df_agg['feature_max'] = df[numeric_cols].max(axis=1)
            df_agg['feature_min'] = df[numeric_cols].min(axis=1)
            df_agg['feature_range'] = df_agg['feature_max'] - df_agg['feature_min']
            
            # Percentile features
            df_agg['feature_p25'] = df[numeric_cols].quantile(0.25, axis=1)
            df_agg['feature_p75'] = df[numeric_cols].quantile(0.75, axis=1)
            df_agg['feature_iqr'] = df_agg['feature_p75'] - df_agg['feature_p25']
            
        self.logger.info("Created aggregate features")
        return df_agg
    
    def create_industry_features(self, df):
        """
        Create industry-specific features for credit risk
        
        Args:
            df (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Dataset with industry-specific features
        """
        df_industry = df.copy()
        
        # Financial health score
        if all(col in df.columns for col in ['debt_to_equity', 'current_ratio', 'return_on_assets']):
            df_industry['financial_health_score'] = (
                -df['debt_to_equity'] * 0.3 +
                df['current_ratio'] * 0.3 +
                df['return_on_assets'] * 0.4
            )
        
        # Liquidity risk score
        if all(col in df.columns for col in ['current_ratio', 'quick_ratio']):
            df_industry['liquidity_risk_score'] = (
                df['current_ratio'] * 0.6 +
                df['quick_ratio'] * 0.4
            )
        
        # Leverage risk score
        if all(col in df.columns for col in ['debt_to_equity', 'debt_to_assets']):
            df_industry['leverage_risk_score'] = (
                df['debt_to_equity'] * 0.5 +
                df['debt_to_assets'] * 0.5
            )
        
        # Profitability score
        if all(col in df.columns for col in ['return_on_assets', 'return_on_equity', 'profit_margin']):
            df_industry['profitability_score'] = (
                df['return_on_assets'] * 0.4 +
                df['return_on_equity'] * 0.3 +
                df['profit_margin'] * 0.3
            )
        
        # Company size categories
        if 'total_assets' in df.columns:
            asset_percentiles = df['total_assets'].quantile([0.33, 0.67])
            df_industry['company_size'] = pd.cut(
                df['total_assets'],
                bins=[-np.inf, asset_percentiles.iloc[0], asset_percentiles.iloc[1], np.inf],
                labels=['Small', 'Medium', 'Large']
            )
        
        self.logger.info("Created industry-specific features")
        return df_industry
    
    def select_features(self, X, y, method='mutual_info', k=20):
        """
        Select most important features for modeling
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
            method (str): Feature selection method ('mutual_info', 'f_classif')
            k (int): Number of features to select
            
        Returns:
            pd.DataFrame: Selected features
        """
        # Ensure all features are numeric for feature selection
        X_numeric = X.select_dtypes(include=[np.number])
        
        if method == 'mutual_info':
            score_func = mutual_info_classif
        elif method == 'f_classif':
            score_func = f_classif
        else:
            raise ValueError("Method must be 'mutual_info' or 'f_classif'")
        
        # Handle the case where k is larger than available features
        k = min(k, X_numeric.shape[1])
        
        self.feature_selector = SelectKBest(score_func=score_func, k=k)
        X_selected = self.feature_selector.fit_transform(X_numeric, y)
        
        # Get selected feature names
        selected_mask = self.feature_selector.get_support()
        selected_features = X_numeric.columns[selected_mask].tolist()
        
        # Create dataframe with selected features
        X_selected_df = pd.DataFrame(
            X_selected, 
            columns=selected_features,
            index=X.index
        )
        
        self.logger.info(f"Selected {k} features using {method}")
        self.logger.info(f"Selected features: {selected_features}")
        
        return X_selected_df
    
    def apply_pca(self, X, n_components=0.95, fit=True):
        """
        Apply Principal Component Analysis for dimensionality reduction
        
        Args:
            X (pd.DataFrame): Features
            n_components (float or int): Number of components or explained variance ratio
            fit (bool): Whether to fit PCA or use existing fit
            
        Returns:
            pd.DataFrame: PCA-transformed features
        """
        if fit or self.pca is None:
            self.pca = PCA(n_components=n_components, random_state=42)
            X_pca = self.pca.fit_transform(X)
        else:
            X_pca = self.pca.transform(X)
        
        # Create feature names for PCA components
        n_components_actual = X_pca.shape[1]
        feature_names = [f'PC{i+1}' for i in range(n_components_actual)]
        
        X_pca_df = pd.DataFrame(X_pca, columns=feature_names, index=X.index)
        
        if fit:
            explained_variance_ratio = self.pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance_ratio)
            self.logger.info(f"PCA reduced features to {n_components_actual} components")
            self.logger.info(f"Explained variance ratio: {cumulative_variance[-1]:.3f}")
        
        return X_pca_df
    
    def feature_engineering_pipeline(self, X, y=None, include_interactions=True, 
                                   include_polynomials=False, include_pca=False,
                                   feature_selection_k=None):
        """
        Complete feature engineering pipeline
        
        Args:
            X (pd.DataFrame): Input features
            y (pd.Series, optional): Target variable for feature selection
            include_interactions (bool): Whether to create interaction features
            include_polynomials (bool): Whether to create polynomial features
            include_pca (bool): Whether to apply PCA
            feature_selection_k (int, optional): Number of features to select
            
        Returns:
            pd.DataFrame: Engineered features
        """
        self.logger.info("Starting feature engineering pipeline...")
        
        X_engineered = X.copy()
        
        # Calculate financial ratios
        X_engineered = self.calculate_financial_ratios(X_engineered)
        
        # Create industry-specific features
        X_engineered = self.create_industry_features(X_engineered)
        
        # Create aggregate features
        X_engineered = self.create_aggregate_features(X_engineered)
        
        # Create interaction features
        if include_interactions:
            X_engineered = self.create_interaction_features(X_engineered)
        
        # Create polynomial features
        if include_polynomials:
            X_engineered = self.create_polynomial_features(X_engineered)
        
        # Feature selection
        if feature_selection_k and y is not None:
            X_engineered = self.select_features(X_engineered, y, k=feature_selection_k)
        
        # Apply PCA
        if include_pca:
            # Only apply PCA to numeric features
            numeric_cols = X_engineered.select_dtypes(include=[np.number]).columns
            X_numeric = X_engineered[numeric_cols]
            X_pca = self.apply_pca(X_numeric)
            
            # Combine with non-numeric features if any
            non_numeric_cols = X_engineered.select_dtypes(exclude=[np.number]).columns
            if len(non_numeric_cols) > 0:
                X_engineered = pd.concat([X_pca, X_engineered[non_numeric_cols]], axis=1)
            else:
                X_engineered = X_pca
        
        self.logger.info(f"Feature engineering completed. Final shape: {X_engineered.shape}")
        
        return X_engineered