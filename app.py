import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import joblib

# Load the trained model and column names used during training
model = joblib.load('/Users/sujankonda/Downloads/Guvi_projects/Cardekho/RandomForestRegressor_model.pkl')
columns_to_match = joblib.load('/Users/sujankonda/Downloads/Guvi_projects/Cardekho/columns_to_match.pkl')  # Saved list of columns used during training

# Streamlit input form
st.title('Car Price Prediction')

# User input fields
vehicle_age = st.number_input('Vehicle Age', min_value=0, max_value=20)
km_driven = st.number_input('Kilometers Driven', min_value=0)
mileage = st.number_input('Mileage')
engine = st.number_input('Engine')
max_power = st.number_input('Max Power')
seats = st.number_input('Seats', min_value=2, max_value=8)

# Categorical feature inputs
seller_type = st.selectbox('Seller Type', ['Dealer', 'Individual', 'Trustmark Dealer'])
fuel_type = st.selectbox('Fuel Type', ['CNG', 'Diesel', 'Electric', 'LPG', 'Petrol'])
transmission_type = st.selectbox('Transmission Type', ['Automatic', 'Manual'])

# Create the input DataFrame from user inputs
user_input = pd.DataFrame({
    'vehicle_age': [vehicle_age],
    'km_driven': [km_driven],
    'mileage': [mileage],
    'engine': [engine],
    'max_power': [max_power],
    'seats': [seats],
    'seller_type': [seller_type],
    'fuel_type': [fuel_type],
    'transmission_type': [transmission_type]
})

# Apply One-Hot Encoding (pd.get_dummies) to categorical variables
user_input_encoded = pd.get_dummies(user_input, drop_first=True)

# Reindex the encoded input to match the training data columns
user_input_encoded = user_input_encoded.reindex(columns=columns_to_match, fill_value=0)

# Predict the car price using the trained model
try:
    predicted_price = model.predict(user_input_encoded)
    st.write(f"Predicted Price: ₹{predicted_price[0]:,.2f}")
except Exception as e:
    st.error(f"An error occurred during prediction: {e}")
