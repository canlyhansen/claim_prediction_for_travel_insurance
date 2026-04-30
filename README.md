<h1> Travel Insurance Claim Prediction & Risk Management using Machine Learning </h1>

## 1. Project Overview
This project develops a machine learning model to predict customer insurance claims and support proactive risk management in the travel insurance business. The goal is to reduce financial uncertainty caused by unexpected claims by identifying high-risk customers before claims occur.

Key Objectives:
- Predict whether a customer will submit an insurance claim (classification problem)
- Minimize financial risk by reducing missed claim cases (False Negatives)
- Translate model performance into measurable business impact
- Support risk-based decision making in underwriting and pricing

## 2. Business Problem
Insurance companies face financial risk when claims occur unexpectedly. Without predictive tools, all claims are treated as unpredictable events, leading to reactive decision-making and potential liquidity issues.

Key challenges:
- Claims are rare (only ~1.7% of customers), creating a highly imbalanced problem
- Missing a claim (False Negative) leads to high financial impact
- Over-predicting claims (False Positive) increases operational cost

This project addresses these challenges by building a model that prioritizes Recall, ensuring high-risk customers are identified early.

## 3. Dataset
- Dataset Name: Travel Insurance Dataset
- Source: Kaggle (Zahier Nasrudin, 2019)
- Size:
  - Before cleaning: 44,328 rows, 11 columns
  - After cleaning: 38,419 rows, 10 columns

Data Description:
The dataset contains customer and policy information, including:
- Customer demographics (e.g., Age)
- Policy details (Product Name, Agency, Distribution Channel)
- Financial variables (Net Sales, Commission)
- Travel characteristics (Duration, Destination)

Data Limitations:
- Data reflects historical patterns (pre-2019)
- No external or behavioral features (e.g., travel conditions, past claims)
- Highly imbalanced dataset (~1.7% claim rate)

## 4. Technologies Used
- Programming Language: Python
- Data Processing & Analysis: Pandas, NumPy
- Data Visualization: Matplotlib, Seaborn
- Machine Learning & Modeling:
  - Scikit-learn (Pipeline, ColumnTransformer, model selection, evaluation metrics)
  - LightGBM, XGBoost
  - Imbalanced-learn (resampling techniques: RandomOverSampler, RandomUnderSampler, SMOTE)
- Feature Engineering & Encoding:
  - OneHotEncoder (Scikit-learn)
  - BinaryEncoder (category_encoders)
  - Feature Selection: SelectPercentile
- Model Evaluation & Optimization:
  - Cross-validation (StratifiedKFold, cross_validate)
  - Hyperparameter tuning (RandomizedSearchCV)
  - Metrics: Recall, Precision, F2-score, PR-AUC
- Development Tools: Jupyter Notebook, Git (version control)

## 5. Methodology
5.1 Data Preprocessing
- Removed duplicates and invalid records (outliers)
- Dropped irrelevant feature (Gender) due to irrelevance
- Applied scaling using RobustScaler
- Encoded categorical variables using: OneHotEncoder (low/moderate cardinality) and BinaryEncoder (high cardinality – Destination)

5.2 Exploratory Data Analysis
Key insights:
- Claims are rare → highly imbalanced problem
- Higher Net Sales, Commission, and Duration increase claim risk
- Claim rates vary across products, agencies, and destinations
- Customer behavior is not random → suitable for predictive modeling

5.3 Modeling Approach
Models evaluated:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- XGBoost
- LightGBM (selected final model)

Evaluation metrics:
- Recall (Primary metric) → minimize missed claims
- PR-AUC (for imbalanced data)
- Precision & F2-score (balance trade-off)

5.4 Final Model
- Model: LightGBM
- Imbalance Handling: scale_pos_weight
- Hyperparameter Tuning: RandomizedSearchCV
- Threshold Optimization: 0.34
- Performance:
  - Recall: ~0.80
  - Precision: ~0.047
  - F2-score: ~0.189
  - PR-AUC: ~0.093

## 6. Business Impact
The model’s predictions are translated into financial outcomes, with key results:
- Missed claims reduced from 133 → 27 (~80% improvement)
- Financial risk from unexpected claims reduced significantly
- Introduces manageable operational cost from false positives
- Total risk reduced by ~$17,600 (7.35%)

## 7. Recommendations
7.1 Business Implementation
- Use model to flag high-risk customers during underwriting and pricing
- Integrate into policy approval and monitoring processes
- Retrain model every 3–6 months

7.2 Data Improvement
- Add customer claim history (e.g., past claims, claim frequency)
- Include external factors (e.g., destination risk, seasonality)
- Improve data completeness and quality

7.3 Model Enhancement
- Optimize threshold based on business cost
- Explore ensemble methods (stacking/blending)
- Extend explainability (e.g., SHAP dependence plots)

## 8. Project Structure

```
├── README.md          <- The top-level README for developers using this project.
|
├── data
│   ├── raw            <- Data from third party sources.
│   └── cleaned        <- The data that has been cleaned.
│
├── notebooks          <- Jupyter Notebook (analysis & modeling)
│
└── model              <- Saved model (pickle file)

```

## 9. Contact
- Name: Canly Hansen Sudirman
- Email: canlyhansen@gmail.com
- Linkedin: https://www.linkedin.com/in/canly-hansen-sudirman-a426298b/
