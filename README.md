# Customer Churn Prediction Using Machine Learning

## Project Overview
A machine learning classification project that predicts whether a telecom customer is likely to leave a company.

## Objectives
- Analyze customer churn patterns.
- Identify factors influencing customer churn.
- Build and compare classification models.

## Dataset
IBM Telco Customer Churn Dataset.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Machine Learning Models
- Logistic Regression
- Random Forest Classifier

## Workflow
1. Data Loading
2. Data Understanding
3. Data Cleaning
4. Exploratory Data Analysis
5. Feature Engineering
6. Data Preprocessing
7. Train-Test Split
8. Model Training
9. Model Evaluation
10. Feature Importance Analysis

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Project Files
- Customer_Churn_Prediction.ipynb
- customer_churn_model.pkl
- model_features.pkl
- requirements.txt
## Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 80.55% | 65.82% | 55.62% | 60.29% |
| Random Forest | 78.64% | 62.37% | 49.20% | 55.01% |

### Best Performing Model
Logistic Regression achieved the highest scores across all four evaluation metrics in this experiment.

### Key Observation
Logistic Regression performed better than Random Forest on the current train-test split. Further improvements can be explored through feature scaling, hyperparameter tuning, and classification threshold optimization.
## Live Demo

Try the deployed application here:

[Customer Churn Prediction App](https://customer-churn-prediction-638sv2ucqr4kb9qideundj.streamlit.app/)
