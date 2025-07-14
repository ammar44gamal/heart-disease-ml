import streamlit as st
import joblib
import numpy as np

# ----------------------------
# Load your model and features
# ----------------------------
model = joblib.load("models/final_model.pkl")
features = joblib.load("models/model_features.pkl")

st.title("🩺 Heart Disease Risk Predictor")

st.write("Enter patient information below:")

# ----------------------------
# Create user input fields
# ----------------------------
user_input = []

for feature in features:
    value = st.number_input(f"{feature}", min_value=0.0, step=1.0)
    user_input.append(value)

# Convert input to numpy array
input_array = np.array(user_input).reshape(1, -1)

# ----------------------------
# Make prediction
# ----------------------------
if st.button("Predict"):
    prediction = model.predict(input_array)[0]
    prob = model.predict_proba(input_array)[0]

    st.subheader("🧠 Prediction Result:")
    st.write(f"Predicted class: {prediction}")
    st.write(f"Probability: {np.max(prob) * 100:.2f}% confidence")