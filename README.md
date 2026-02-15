# Machine Learning Classification Deployment -- Assignment 2

## a. Problem Statement

The objective of this assignment is to implement multiple machine
learning classification models on a real-world dataset and deploy them
using a Streamlit web application.

The task involves building six different classification models,
evaluating them using multiple performance metrics, comparing their
performance, and providing an interactive user interface for
predictions.

The final application is deployed on Streamlit Community Cloud to
demonstrate an end-to-end ML workflow including model training,
evaluation, UI design, and cloud deployment.

------------------------------------------------------------------------

## b. Dataset Description

**Dataset Name:** Breast Cancer Wisconsin Dataset
**Source:** Scikit-learn Built-in Dataset (Original UCI Repository)

**Dataset Characteristics:** - Number of Instances: 569
- Number of Features: 30 numerical features
- Target Variable: Binary Classification
- 0 → Malignant
- 1 → Benign

The dataset contains diagnostic measurements computed from digitized
images of breast mass samples.

------------------------------------------------------------------------

## c. Models Used and Evaluation Metrics

The following six machine learning models were implemented on the same
dataset:

1.  Logistic Regression
2.  Decision Tree Classifier
3.  K-Nearest Neighbors (KNN)
4.  Naive Bayes (Gaussian)
5.  Random Forest (Ensemble Model)
6.  XGBoost (Ensemble Model)

------------------------------------------------------------------------

## Evaluation Metrics Used

1.  Accuracy
2.  AUC Score
3.  Precision
4.  Recall
5.  F1 Score
6.  Matthews Correlation Coefficient (MCC Score)

------------------------------------------------------------------------

## Model Comparison Table

  | ML Model Name               | Accuracy    | AUC     | Precision    | Recall    | F1 Score    | MCC| 
  | --------------------------  | ----------  | ------  | -----------  | --------  | ----------  | ------| 
  | Logistic Regression         | 0.97        | 0.99    | 0.98         | 0.96      | 0.97        | 0.94| 
  | Decision Tree               | 0.93        | 0.92    | 0.94         | 0.91      | 0.92        | 0.86| 
  | KNN                         | 0.95        | 0.97   | 0.96         | 0.94      | 0.95        | 0.90| 
  | Naive Bayes                 | 0.92       | 0.96    | 0.93         | 0.90      | 0.91        | 0.84| 
  | Random Forest (Ensemble)    | 0.98        | 0.99    | 0.98         | 0.98      | 0.98        | 0.96| 
  | XGBoost (Ensemble)          | 0.98        | 0.99    | 0.99         | 0.97     |  0.98        | 0.96| 

------------------------------------------------------------------------

## Observations on Model Performance

  -----------------------------------------------------------------------
  | ML Model Name |        Observation About Model Performance |
  | -------------------- | -------------------------------------------------- |
  | Logistic Regression |  Performs strongly due to near linear separability of the dataset. |
  | Decision Tree |        Slight overfitting observed due to high variance. |
  | KNN |                  Good accuracy but slower for large datasets. |
  | Naive Bayes |          Reasonable performance; independence assumption limits accuracy. |
  | Random Forest |        High stability and strong generalization performance. |
  | XGBoost |              Best overall performance with robust predictive power. |
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## Streamlit Application Features

-   CSV dataset upload option
-   Model selection dropdown
-   Display of evaluation metrics
-   Confusion Matrix
-   Classification Report

------------------------------------------------------------------------

## Project Structure

project-folder/ │-- app.py\
│-- requirements.txt\
│-- README.md\
│-- model/

------------------------------------------------------------------------

## Conclusion

Ensemble methods such as Random Forest and XGBoost achieved the best
performance due to their ability to reduce variance and improve
generalization.
