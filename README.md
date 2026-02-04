# Bank Customer Churn Prediction

This project predicts whether a bank customer will churn (exit) based on demographic and account data from a structured CSV dataset.[file:2]

## 1. Overview

- Dataset: 10,000 customers with 13 columns from `Churn.csv`.[file:2]  
- Target: `Exited` (0 = stay, 1 = exit).[file:2]  
- Objective: Build and compare classification models to support churn‑reduction actions (e.g., which customers to target with retention offers).[file:2]

## 2. Data Description

Columns after cleaning:[file:2]

- Removed ID‑like fields: `CustomerId`, `Surname`.[file:2]  
- Categorical features: `Geography` (France, Germany, Spain), `Gender` (Female, Male).[file:2]  
- Numerical / binary features: `CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`.[file:2]  

Data quality checks confirmed 10,000 rows, 13 columns, and no missing values.[file:2]

Exploratory analysis included:[file:2]

- Correlation examples (e.g., Spearman correlation between `IsActiveMember` and `Exited`).[file:2]  
- Distribution plots such as age distribution by churn and bar charts for churn by geography and by activity status.[file:2]

## 3. Preprocessing

Steps applied before modeling:[file:2]

- Encoded `Gender` and `Geography` using label encoders to convert categories to numeric values.[file:2]  
- Split data into features (`X`, all columns except `Exited`) and target (`y = Exited`).[file:2]  
- Converted feature matrix to a pure float array for compatibility with scikit‑learn.[file:2]  
- Scaled features (standardization) to center and normalize the numeric inputs.[file:2]

## 4. Models and Evaluation

Models used:[file:2]

- K‑Nearest Neighbors (KNN) classifier with hyperparameter tuning via GridSearchCV.[file:2]  
- Decision Tree classifier (criterion entropy, limited depth) on the same scaled train/test split.[file:2]  

Training and evaluation:[file:2]

- Train/test split examples:
  - 70/30 split for KNN hyperparameter tuning.[file:2]  
  - 80/20 split for decision tree evaluation.[file:2]  
- Reported metrics:
  - Best K for KNN (from grid search) and its cross‑validation score.[file:2]  
  - KNN test MSE.[file:2]  
  - Decision Tree test MSE.[file:2]  

(You can extend this to add classification metrics such as accuracy, precision, recall, F1‑score, and ROC‑AUC.)

## 5. Example Inference

The notebook demonstrates how to score a single new customer:[file:2]

1. Define a new customer profile with values for credit score, geography, gender, age, tenure, balance, number of products, credit card status, activity status, and salary.[file:2]  
2. Apply the same label encoders to map `Gender` and `Geography` to numeric values.[file:2]  
3. Apply the same scaling transformation used for the training data (using the training mean and standard deviation).[file:2]  
4. Feed the scaled feature vector into the trained decision tree to obtain a prediction: 0 (stay) or 1 (exit).[file:2]  

This shows how the model can be used in production for real‑time churn risk scoring.[file:2]

## 6. How to Use This Project

- Requirements: Python environment with pandas, numpy, seaborn, matplotlib, scikit‑learn, and scipy installed.[file:2]  
- Files: Place `Churn.csv` in the same directory as the notebook.[file:2]  
- Workflow:
  - Run the notebook to load and clean the data.  
  - Encode and scale features.  
  - Train KNN and Decision Tree models.  
  - Inspect evaluation metrics and visualizations (correlations, distributions, churn rate plots).  
  - Use the example section to test predictions for new customers.[file:2]

## 7. Possible Extensions

Future improvements you can add:[file:2]

- Replace or complement MSE with classification metrics (accuracy, precision, recall, F1, ROC‑AUC).[file:2]  
- Perform hyperparameter tuning with grid search for additional models (e.g., Random Forest, Gradient Boosting).[file:2]  
- Address class imbalance if present (class weights, oversampling/undersampling).  
- Persist trained models (e.g., via joblib) and expose them via an API or a Streamlit web app for interactive use.[file:2]
