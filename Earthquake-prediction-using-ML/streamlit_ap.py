import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Title of the app
st.title('🌍 Earthquake Prediction Using Machine Learning')

# Load model and scaler (cached)
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

# Load the model and scaler
model, scaler = load_model_and_scaler()

# Input fields for prediction
st.write("## 🌐 Input Features for Prediction")
latitude = st.number_input('Latitude:', value=0.0)
longitude = st.number_input('Longitude:', value=0.0)
depth = st.number_input('Depth (km):', value=0.0)

# Predict button
if st.button('🔍 Predict Magnitude'):
    input_df = pd.DataFrame([[latitude, longitude, depth]], columns=['latitude', 'longitude', 'depth'])
    input_scaled = scaler.transform(input_df)
    predicted_mag = model.predict(input_scaled)[0]
    st.success(f'The predicted magnitude of the earthquake using optimized XGBoost is: **{predicted_mag:}**')
