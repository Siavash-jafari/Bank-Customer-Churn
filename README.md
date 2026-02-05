# Bank Customer Churn Prediction

This project predicts whether a bank customer will churn (exit) using demographic and account‑level data from a structured CSV dataset.

## 1. Overview

- Dataset: 10,000 customers with 13 columns in `Churn.csv`.
- Target variable: `Exited` (0 = stay, 1 = exit).
- Goal: Train and compare classification models to support churn‑reduction actions, such as identifying customers to target with retention offers.

## 2. Data Description

After basic cleaning, the dataset contains the following features:

- Dropped ID‑like fields: `CustomerId`, `Surname`, which do not carry predictive information.
- Categorical features: `Geography` (France, Germany, Spain), `Gender` (Female, Male).
- Numerical/binary features: `CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`.

Data quality checks confirm 10,000 rows, 13 columns, and no missing values.

Exploratory analysis includes:

- Correlation analysis (for example, Spearman correlation between `IsActiveMember` and `Exited`).
- Distribution plots such as age distributions split by churn status.
- Bar charts of churn rate by geography and by activity status.

## 3. Preprocessing

Before modeling, the following preprocessing steps are applied:

- Encode `Gender` and `Geography` using label encoders to convert categories to numeric codes.
- Split the data into features `X` (all columns except `Exited`) and target `y = Exited`.
- Cast the feature matrix to a float array for compatibility with scikit‑learn estimators.
- Standardize features (zero mean, unit variance) so that distance‑based models like KNN work properly.

## 4. Models and Evaluation

Two baseline classification models are implemented using the same preprocessed data:

- K‑Nearest Neighbors (KNN) classifier with hyperparameter tuning via `GridSearchCV` to find the best number of neighbors.
- Decision Tree classifier (entropy criterion, limited depth to reduce overfitting).

Training and evaluation setup:

- Example train/test splits:
  - 70/30 split for KNN hyperparameter tuning and model selection.
  - 80/20 split for Decision Tree evaluation.
- Reported metrics include:
  - Best `k` value for KNN from grid search and its cross‑validation score.
  - Test mean squared error (MSE) for KNN.
  - Test MSE for the Decision Tree.

You can extend the notebook to report additional classification metrics, such as accuracy, precision, recall, F1‑score, confusion matrix, and ROC‑AUC.

## 5. Example Inference

The notebook shows how to generate a churn prediction for a single new customer:

1. Define a new customer profile with values for credit score, geography, gender, age, tenure, balance, number of products, credit card status, activity status, and estimated salary.
2. Apply the fitted label encoders to map `Gender` and `Geography` to the same numeric codes used during training.
3. Apply the fitted scaler (using training mean and standard deviation) to transform the feature vector.
4. Feed the scaled vector into the trained Decision Tree model to obtain a prediction: 0 (stay) or 1 (exit).

This workflow demonstrates how the model can be used in production for real‑time churn risk scoring.

## 6. How to Use

### Requirements

Install the following Python packages:

- `pandas`, `numpy` for data handling.
- `seaborn`, `matplotlib` for visualization.
- `scikit-learn`, `scipy` for modeling and evaluation.

### Files

- `Churn.csv`: main dataset containing 10,000 customer records.
- Jupyter notebook: contains data exploration, preprocessing, model training, evaluation, and inference examples.

Place `Churn.csv` in the same directory as the notebook.

### Workflow

- Run the notebook to:
  - Load and clean the data.
  - Encode and scale features.
  - Train KNN and Decision Tree models.
  - Inspect evaluation metrics and visualizations (correlations, distributions, churn rate plots).
  - Use the example section to test predictions for new customer profiles.
