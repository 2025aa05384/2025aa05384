import joblib
import pandas as pd
import streamlit as st

st.title("ML Classification Deployment App")

file = st.file_uploader("Upload Test CSV", type=["csv"])

model_choice = st.selectbox(
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

    # Load saved feature names and scaler
    feature_names = joblib.load("model/feature_names.pkl")
    scaler = joblib.load("model/scaler.pkl")

    # Drop target if exists
    if "target" in data.columns:
        data = data.drop("target", axis=1)

    # Ensure correct column order
    missing_cols = set(feature_names) - set(data.columns)

    if missing_cols:
        st.error(f"Uploaded file is missing required columns: {missing_cols}")
        st.stop()

    extra_cols = set(data.columns) - set(feature_names)
    if extra_cols:
        data = data.drop(columns=extra_cols)

    data = data[feature_names]


    # Apply scaling
    data_scaled = scaler.transform(data)

    # Load model
    model = joblib.load(f"model/{model_choice}.pkl")

    predictions = model.predict(data_scaled)

    st.write("Predictions:")
    st.write(predictions)
