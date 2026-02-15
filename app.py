import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

st.title("ML Classification Deployment App")

st.sidebar.header("Upload Test Dataset (CSV)")
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Logistic Regression",
     "Decision Tree",
     "KNN",
     "Naive Bayes",
     "Random Forest",
     "XGBoost"]
)

if file:
    data = pd.read_csv(file)
    st.write("Preview of Uploaded Data")
    st.dataframe(data.head())

    model = joblib.load(f"model/{model_choice}.pkl")

    predictions = model.predict(data)
    st.write("Predictions:")
    st.write(predictions)

    if st.checkbox("Show Confusion Matrix"):
        st.write(confusion_matrix(predictions, predictions))

    if st.checkbox("Show Classification Report"):
        st.text(classification_report(predictions, predictions))
