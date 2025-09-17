# Credit Risk Prediction Model - Capstone Project

This project aims to build a comprehensive credit risk prediction model that incorporates multiple variables to assess the default probability and credit risk of companies. The model uses advanced machine learning techniques and feature engineering to provide accurate risk assessments for lending decisions.

## 🎯 Project Overview

Credit risk assessment is crucial for financial institutions to make informed lending decisions. This project develops a multi-variable predictive model that:

- Predicts the probability of company default
- Assesses overall credit risk
- Incorporates multiple financial and business variables
- Provides interpretable results for business decisions
- Offers comprehensive model evaluation and comparison

## 🏗️ Project Structure

```
capstone_project_dpm/
├── src/
│   └── credit_risk_model/
│       ├── __init__.py
│       ├── data_preprocessing.py    # Data loading and preprocessing
│       ├── feature_engineering.py   # Advanced feature engineering
│       ├── model_trainer.py         # Model training and optimization
│       └── model_evaluator.py       # Model evaluation and visualization
├── data/
│   ├── raw/                         # Raw datasets
│   └── processed/                   # Processed datasets
├── notebooks/                       # Jupyter notebooks for analysis
├── tests/                          # Unit tests
├── config/                         # Configuration files
├── models/                         # Saved trained models
├── outputs/                        # Results and visualizations
├── main.py                         # Main application script
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

## 🚀 Features

### Data Preprocessing
- **Multi-format support**: CSV, Excel files
- **Missing value handling**: Multiple strategies (median, mean, mode, drop)
- **Outlier detection**: IQR and Z-score methods
- **Categorical encoding**: Label encoding for categorical variables
- **Feature scaling**: StandardScaler for numerical features
- **Sample dataset generation**: Creates realistic credit risk datasets

### Feature Engineering
- **Financial ratios**: Liquidity, leverage, profitability, efficiency ratios
- **Interaction features**: Multiplicative and ratio interactions
- **Polynomial features**: Non-linear relationship modeling
- **Aggregate features**: Statistical summaries and percentiles
- **Industry-specific features**: Financial health scores, risk indicators
- **Feature selection**: Mutual information and F-test based selection
- **Dimensionality reduction**: PCA for high-dimensional datasets

### Machine Learning Models
- **Logistic Regression**: Linear baseline model
- **Random Forest**: Ensemble tree-based model
- **Gradient Boosting**: Advanced gradient boosting
- **XGBoost**: Extreme gradient boosting
- **LightGBM**: Fast gradient boosting
- **Support Vector Machine**: Non-linear classification
- **Ensemble methods**: Voting and weighted averaging

### Model Evaluation
- **Classification metrics**: Accuracy, Precision, Recall, F1-score, AUC
- **Business metrics**: Profit analysis, opportunity cost, approval rates
- **Visualizations**: ROC curves, PR curves, confusion matrices
- **Calibration analysis**: Probability calibration assessment
- **Feature importance**: Model interpretability analysis
- **Model comparison**: Side-by-side performance comparison

### Advanced Capabilities
- **Class imbalance handling**: SMOTE, undersampling, SMOTEENN
- **Hyperparameter optimization**: Grid search with cross-validation
- **Model persistence**: Save and load trained models
- **Comprehensive logging**: Detailed workflow tracking
- **Automated reporting**: Generated evaluation reports

## 📋 Requirements

### Python Dependencies
```
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
matplotlib>=3.5.0
seaborn>=0.11.0
xgboost>=1.6.0
lightgbm>=3.3.0
imbalanced-learn>=0.9.0
joblib>=1.1.0
jupyter>=1.0.0
```

### System Requirements
- Python 3.8+
- 4GB+ RAM (for large datasets)
- 1GB free disk space

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone https://github.com/NareshChethala/capstone_project_dpm.git
cd capstone_project_dpm
```

2. **Create virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Running the Complete Workflow

Execute the main script to run the entire credit risk modeling pipeline:

```bash
python main.py
```

This will:
1. Create a sample dataset (or load your data)
2. Preprocess the data
3. Engineer features
4. Train multiple models
5. Evaluate and compare models
6. Generate visualizations and reports
7. Save results and models

### Using Individual Components

```python
from src.credit_risk_model import DataPreprocessor, FeatureEngineer, ModelTrainer, ModelEvaluator

# Data preprocessing
preprocessor = DataPreprocessor()
X, y = preprocessor.create_sample_dataset(n_samples=1000)
X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(X, y)

# Feature engineering
engineer = FeatureEngineer()
X_train_eng = engineer.feature_engineering_pipeline(X_train, y_train)

# Model training
trainer = ModelTrainer()
models = trainer.train_all_models(X_train_eng, y_train)

# Model evaluation
evaluator = ModelEvaluator()
results = evaluator.compare_models(models, X_test, y_test)
```

## 📊 Usage Examples

### 1. Loading Your Own Data

```python
from src.credit_risk_model import DataPreprocessor

preprocessor = DataPreprocessor()
X, y = preprocessor.load_data('your_credit_data.csv', target_column='default')
```

### 2. Custom Feature Engineering

```python
from src.credit_risk_model import FeatureEngineer

engineer = FeatureEngineer()
X_engineered = engineer.feature_engineering_pipeline(
    X, y,
    include_interactions=True,
    include_polynomials=True,
    feature_selection_k=25
)
```

### 3. Training Specific Models

```python
from src.credit_risk_model import ModelTrainer

trainer = ModelTrainer()
models = trainer.train_all_models(
    X_train, y_train,
    models_to_train=['xgboost', 'lightgbm', 'random_forest'],
    use_grid_search=True,
    handle_imbalance='smote'
)
```

### 4. Business Impact Analysis

```python
from src.credit_risk_model import ModelEvaluator

evaluator = ModelEvaluator()
business_metrics = evaluator.calculate_business_metrics(
    y_true, y_pred, y_prob,
    loan_amounts=loan_amounts,
    default_loss_rate=0.6
)
```

## 📈 Model Performance

The framework supports comprehensive performance evaluation:

### Classification Metrics
- **Accuracy**: Overall prediction accuracy
- **Precision**: Ratio of correct positive predictions
- **Recall (Sensitivity)**: Ratio of actual positives identified
- **Specificity**: Ratio of actual negatives identified
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve

### Business Metrics
- **Net Profit**: Total profit considering revenues and losses
- **Profit Margin**: Profit as percentage of total loans
- **Approval Rate**: Percentage of loans approved
- **Loss Rate**: Actual default rate among approved loans
- **Opportunity Cost**: Revenue lost from rejected good customers

## 🔧 Configuration

### Model Parameters

Customize model parameters in the training script:

```python
# Hyperparameter grids
param_grids = {
    'xgboost': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
}
```

### Feature Engineering Options

```python
# Feature engineering configuration
engineer.feature_engineering_pipeline(
    X, y,
    include_interactions=True,      # Create interaction features
    include_polynomials=False,      # Add polynomial features
    include_pca=False,              # Apply PCA
    feature_selection_k=20          # Select top K features
)
```

## 📁 Output Files

After running the pipeline, you'll find:

### Models
- `models/credit_risk_model_*.joblib`: Trained model files

### Results
- `outputs/model_comparison_results.csv`: Model performance comparison
- `outputs/feature_importance.csv`: Feature importance rankings
- `outputs/sample_predictions.csv`: Sample predictions with probabilities

### Visualizations
- `outputs/model_comparison.png`: Multi-model performance comparison
- `outputs/best_model_analysis.png`: Detailed best model analysis
- `outputs/feature_importance.png`: Feature importance visualization

### Logs
- `credit_risk_model.log`: Detailed execution logs

## 🧪 Testing

Run the test suite:

```bash
python -m pytest tests/ -v
```

## 📝 Data Requirements

For optimal performance, your dataset should include:

### Required Columns
- **Target variable**: Binary indicator (0 = no default, 1 = default)
- **Financial metrics**: Assets, liabilities, revenue, income
- **Ratios**: Debt-to-equity, current ratio, profit margins

### Recommended Columns
- **Company information**: Age, size, industry
- **Market data**: Stock price, market cap
- **Credit history**: Previous defaults, credit rating

### Sample Data Format
```csv
company_id,total_assets,total_liabilities,revenue,net_income,current_ratio,debt_to_equity,default
1,1000000,600000,500000,50000,1.5,1.5,0
2,2000000,1800000,800000,-20000,0.8,9.0,1
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Naresh Chethala**
- GitHub: [@NareshChethala](https://github.com/NareshChethala)

## 🙏 Acknowledgments

- Scikit-learn community for machine learning tools
- XGBoost and LightGBM teams for gradient boosting implementations
- Imbalanced-learn for handling class imbalance
- The open-source community for inspiration and tools

## 📚 References

1. **Credit Risk Modeling**: Naeem Siddiqi - Credit Risk Scorecards
2. **Machine Learning**: Hands-On Machine Learning by Aurélien Géron
3. **Financial Risk Management**: John Hull - Risk Management and Financial Institutions
4. **Feature Engineering**: Feature Engineering for Machine Learning by Alice Zheng

---

**Note**: This is a capstone project for educational and demonstration purposes. For production use in financial institutions, ensure compliance with regulatory requirements and thorough validation of model performance.