"""
Model Evaluation Module for Credit Risk Modeling

This module provides comprehensive evaluation metrics and visualization tools
for credit risk prediction models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
from sklearn.calibration import calibration_curve
import logging

class ModelEvaluator:
    """
    Comprehensive model evaluation for credit risk models including:
    - Standard classification metrics
    - ROC and PR curves
    - Calibration analysis
    - Feature importance analysis
    - Business metrics (profit/loss analysis)
    - Model comparison and visualization
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.evaluation_results = {}
        
    def calculate_classification_metrics(self, y_true, y_pred, y_prob=None):
        """
        Calculate comprehensive classification metrics
        
        Args:
            y_true (array-like): True labels
            y_pred (array-like): Predicted labels
            y_prob (array-like, optional): Predicted probabilities
            
        Returns:
            dict: Dictionary of classification metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred)
        metrics['recall'] = recall_score(y_true, y_pred)
        metrics['f1_score'] = f1_score(y_true, y_pred)
        
        # Specificity (True Negative Rate)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # AUC if probabilities are provided
        if y_prob is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        
        # Confusion matrix components
        metrics['true_positives'] = tp
        metrics['true_negatives'] = tn
        metrics['false_positives'] = fp
        metrics['false_negatives'] = fn
        
        return metrics
    
    def plot_roc_curve(self, y_true, y_prob, model_name="Model", ax=None):
        """
        Plot ROC curve
        
        Args:
            y_true (array-like): True labels
            y_prob (array-like): Predicted probabilities
            model_name (str): Name of the model for the plot
            ax (matplotlib.axes, optional): Existing axes to plot on
            
        Returns:
            tuple: (fpr, tpr, auc_score)
        """
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = roc_auc_score(y_true, y_prob)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc_score:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fpr, tpr, auc_score
    
    def plot_precision_recall_curve(self, y_true, y_prob, model_name="Model", ax=None):
        """
        Plot Precision-Recall curve
        
        Args:
            y_true (array-like): True labels
            y_prob (array-like): Predicted probabilities
            model_name (str): Name of the model for the plot
            ax (matplotlib.axes, optional): Existing axes to plot on
            
        Returns:
            tuple: (precision, recall, average_precision)
        """
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        average_precision = np.trapz(precision, recall)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(recall, precision, linewidth=2, 
               label=f'{model_name} (AP = {average_precision:.3f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return precision, recall, average_precision
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name="Model", ax=None):
        """
        Plot confusion matrix heatmap
        
        Args:
            y_true (array-like): True labels
            y_pred (array-like): Predicted labels
            model_name (str): Name of the model for the plot
            ax (matplotlib.axes, optional): Existing axes to plot on
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['No Default', 'Default'],
                   yticklabels=['No Default', 'Default'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix - {model_name}')
    
    def plot_calibration_curve(self, y_true, y_prob, model_name="Model", n_bins=10, ax=None):
        """
        Plot calibration curve to assess probability calibration
        
        Args:
            y_true (array-like): True labels
            y_prob (array-like): Predicted probabilities
            model_name (str): Name of the model for the plot
            n_bins (int): Number of bins for calibration
            ax (matplotlib.axes, optional): Existing axes to plot on
            
        Returns:
            tuple: (fraction_of_positives, mean_predicted_value)
        """
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=n_bins
        )
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(mean_predicted_value, fraction_of_positives, "s-", 
               linewidth=2, label=model_name)
        ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title('Calibration Plot')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fraction_of_positives, mean_predicted_value
    
    def calculate_business_metrics(self, y_true, y_pred, y_prob, 
                                 loan_amounts=None, default_loss_rate=0.6):
        """
        Calculate business-specific metrics for credit risk
        
        Args:
            y_true (array-like): True labels (1 = default)
            y_pred (array-like): Predicted labels
            y_prob (array-like): Predicted probabilities
            loan_amounts (array-like, optional): Loan amounts for each case
            default_loss_rate (float): Percentage of loan lost when default occurs
            
        Returns:
            dict: Business metrics
        """
        if loan_amounts is None:
            loan_amounts = np.ones(len(y_true))  # Assume unit loans
        
        loan_amounts = np.array(loan_amounts)
        
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Business outcomes
        # True Negatives: Correctly identified good customers (profit)
        # False Positives: Rejected good customers (opportunity cost)
        # True Positives: Correctly identified bad customers (avoided loss)
        # False Negatives: Approved bad customers (actual loss)
        
        # Calculate profits and losses
        profit_per_good_loan = 0.05  # 5% profit margin on good loans
        
        # Actual defaults among approved loans
        approved_mask = y_pred == 0  # Predicted as non-default
        approved_loans = loan_amounts[approved_mask]
        actual_defaults_approved = y_true[approved_mask]
        
        # Revenue from good approved loans
        good_approved = approved_loans[actual_defaults_approved == 0]
        total_revenue = np.sum(good_approved) * profit_per_good_loan
        
        # Losses from bad approved loans (false negatives)
        bad_approved = approved_loans[actual_defaults_approved == 1]
        total_losses = np.sum(bad_approved) * default_loss_rate
        
        # Net profit
        net_profit = total_revenue - total_losses
        
        # Opportunity cost (good customers rejected)
        rejected_mask = y_pred == 1  # Predicted as default
        rejected_loans = loan_amounts[rejected_mask]
        actual_good_rejected = y_true[rejected_mask]
        good_rejected = rejected_loans[actual_good_rejected == 0]
        opportunity_cost = np.sum(good_rejected) * profit_per_good_loan
        
        business_metrics = {
            'total_revenue': total_revenue,
            'total_losses': total_losses,
            'net_profit': net_profit,
            'opportunity_cost': opportunity_cost,
            'net_profit_after_opportunity_cost': net_profit - opportunity_cost,
            'profit_margin': net_profit / np.sum(loan_amounts) if np.sum(loan_amounts) > 0 else 0,
            'loss_rate_actual': np.sum(bad_approved) / np.sum(approved_loans) if np.sum(approved_loans) > 0 else 0,
            'approval_rate': np.sum(approved_mask) / len(y_pred),
            'precision_weighted': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall_weighted': tp / (tp + fn) if (tp + fn) > 0 else 0
        }
        
        return business_metrics
    
    def evaluate_single_model(self, model, X_test, y_test, model_name="Model", 
                            loan_amounts=None):
        """
        Comprehensive evaluation of a single model
        
        Args:
            model: Trained model
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test labels
            model_name (str): Name of the model
            loan_amounts (array-like, optional): Loan amounts for business metrics
            
        Returns:
            dict: Comprehensive evaluation results
        """
        self.logger.info(f"Evaluating model: {model_name}")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        classification_metrics = self.calculate_classification_metrics(y_test, y_pred, y_prob)
        business_metrics = self.calculate_business_metrics(y_test, y_pred, y_prob, loan_amounts)
        
        # Combine all metrics
        evaluation_result = {
            'model_name': model_name,
            'classification_metrics': classification_metrics,
            'business_metrics': business_metrics,
            'predictions': y_pred,
            'probabilities': y_prob
        }
        
        self.evaluation_results[model_name] = evaluation_result
        
        # Print summary
        self.logger.info(f"Model: {model_name}")
        self.logger.info(f"Accuracy: {classification_metrics['accuracy']:.4f}")
        self.logger.info(f"Precision: {classification_metrics['precision']:.4f}")
        self.logger.info(f"Recall: {classification_metrics['recall']:.4f}")
        self.logger.info(f"F1-Score: {classification_metrics['f1_score']:.4f}")
        self.logger.info(f"ROC-AUC: {classification_metrics['roc_auc']:.4f}")
        self.logger.info(f"Net Profit: ${business_metrics['net_profit']:,.2f}")
        
        return evaluation_result
    
    def compare_models(self, models, X_test, y_test, loan_amounts=None):
        """
        Compare multiple models side by side
        
        Args:
            models (dict): Dictionary of trained models
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test labels
            loan_amounts (array-like, optional): Loan amounts for business metrics
            
        Returns:
            pd.DataFrame: Comparison table of model performance
        """
        comparison_data = []
        
        for model_name, model in models.items():
            evaluation_result = self.evaluate_single_model(
                model, X_test, y_test, model_name, loan_amounts
            )
            
            # Extract key metrics for comparison
            metrics = evaluation_result['classification_metrics']
            business = evaluation_result['business_metrics']
            
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1_Score': metrics['f1_score'],
                'ROC_AUC': metrics['roc_auc'],
                'Specificity': metrics['specificity'],
                'Net_Profit': business['net_profit'],
                'Profit_Margin': business['profit_margin'],
                'Approval_Rate': business['approval_rate']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('ROC_AUC', ascending=False)
        
        return comparison_df
    
    def plot_model_comparison(self, models, X_test, y_test):
        """
        Create comprehensive visualization comparing multiple models
        
        Args:
            models (dict): Dictionary of trained models
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test labels
        """
        n_models = len(models)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # ROC Curves
        ax_roc = axes[0, 0]
        # Precision-Recall Curves
        ax_pr = axes[0, 1]
        # Feature Importance (for first model only)
        ax_feat = axes[1, 0]
        # Metrics Comparison
        ax_metrics = axes[1, 1]
        
        model_metrics = []
        
        for model_name, model in models.items():
            # Make predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            # Plot ROC curve
            self.plot_roc_curve(y_test, y_prob, model_name, ax_roc)
            
            # Plot PR curve
            self.plot_precision_recall_curve(y_test, y_prob, model_name, ax_pr)
            
            # Collect metrics for comparison
            metrics = self.calculate_classification_metrics(y_test, y_pred, y_prob)
            model_metrics.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1_Score': metrics['f1_score'],
                'ROC_AUC': metrics['roc_auc']
            })
        
        # Plot feature importance for the first model (if available)
        first_model = list(models.values())[0]
        if hasattr(first_model, 'feature_importances_'):
            importance = first_model.feature_importances_
            feature_names = X_test.columns
            
            # Get top 10 features
            indices = np.argsort(importance)[-10:]
            ax_feat.barh(range(len(indices)), importance[indices])
            ax_feat.set_yticks(range(len(indices)))
            ax_feat.set_yticklabels([feature_names[i] for i in indices])
            ax_feat.set_xlabel('Feature Importance')
            ax_feat.set_title(f'Top 10 Features - {list(models.keys())[0]}')
        else:
            ax_feat.text(0.5, 0.5, 'Feature importance\nnot available', 
                        ha='center', va='center', transform=ax_feat.transAxes)
            ax_feat.set_title('Feature Importance')
        
        # Plot metrics comparison
        metrics_df = pd.DataFrame(model_metrics)
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC']
        
        x = np.arange(len(models))
        width = 0.15
        
        for i, metric in enumerate(metrics_to_plot):
            ax_metrics.bar(x + i * width, metrics_df[metric], width, label=metric)
        
        ax_metrics.set_xlabel('Models')
        ax_metrics.set_ylabel('Score')
        ax_metrics.set_title('Model Performance Comparison')
        ax_metrics.set_xticks(x + width * 2)
        ax_metrics.set_xticklabels(metrics_df['Model'], rotation=45)
        ax_metrics.legend()
        ax_metrics.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def generate_evaluation_report(self, model_name):
        """
        Generate a comprehensive text report for a model
        
        Args:
            model_name (str): Name of the model to report on
            
        Returns:
            str: Formatted evaluation report
        """
        if model_name not in self.evaluation_results:
            return f"No evaluation results found for model: {model_name}"
        
        result = self.evaluation_results[model_name]
        cm = result['classification_metrics']
        bm = result['business_metrics']
        
        report = f"""
        ================================
        MODEL EVALUATION REPORT
        ================================
        Model: {model_name}
        
        CLASSIFICATION METRICS:
        -----------------------
        Accuracy:     {cm['accuracy']:.4f}
        Precision:    {cm['precision']:.4f}
        Recall:       {cm['recall']:.4f}
        F1-Score:     {cm['f1_score']:.4f}
        Specificity:  {cm['specificity']:.4f}
        ROC-AUC:      {cm['roc_auc']:.4f}
        
        CONFUSION MATRIX:
        -----------------
        True Positives:  {cm['true_positives']}
        True Negatives:  {cm['true_negatives']}
        False Positives: {cm['false_positives']}
        False Negatives: {cm['false_negatives']}
        
        BUSINESS METRICS:
        -----------------
        Net Profit:         ${bm['net_profit']:,.2f}
        Total Revenue:      ${bm['total_revenue']:,.2f}
        Total Losses:       ${bm['total_losses']:,.2f}
        Opportunity Cost:   ${bm['opportunity_cost']:,.2f}
        Profit Margin:      {bm['profit_margin']:.2%}
        Approval Rate:      {bm['approval_rate']:.2%}
        Actual Loss Rate:   {bm['loss_rate_actual']:.2%}
        
        ================================
        """
        
        return report