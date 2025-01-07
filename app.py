import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


# Streamlit app
st.title('Customer Churn Prediction')

# User input with placeholders
geography = st.selectbox('Geography (Select)', onehot_encoder_geo.categories_[0], index=0)
gender = st.selectbox('Gender (Select)', label_encoder_gender.classes_, index=0)
age = st.slider('Age', 18, 92, value=30)  # Default age set to 30
balance = st.text_input('Balance', placeholder='Enter balance amount')  # Placeholder for balance
credit_score = st.text_input('Credit Score', placeholder='Enter credit score')  # Placeholder for credit score
estimated_salary = st.text_input('Estimated Salary', placeholder='Enter estimated salary')  # Placeholder for salary
tenure = st.slider('Tenure (Years)', 0, 10, value=5)  # Default tenure set to 5
num_of_products = st.slider('Number of Products', 1, 4, value=1)
has_cr_card = st.selectbox('Has Credit Card', ['No', 'Yes'], index=1)  # Using Yes/No options
is_active_member = st.selectbox('Is Active Member', ['No', 'Yes'], index=1)  # Using Yes/No options

# Convert text inputs to numeric (if not empty)
balance = float(balance) if balance else 0.0
credit_score = int(credit_score) if credit_score else 0
estimated_salary = float(estimated_salary) if estimated_salary else 0.0

# Convert Yes/No inputs to numeric
has_cr_card_numeric = 1 if has_cr_card == 'Yes' else 0
is_active_member_numeric = 1 if is_active_member == 'Yes' else 0

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card_numeric],
    'IsActiveMember': [is_active_member_numeric],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')
