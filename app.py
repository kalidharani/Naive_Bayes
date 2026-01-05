import streamlit as st
import pandas as pd
import joblib

# 1. Load the model and feature names
# Make sure 'titanic_gnb_model.pkl' is in the same folder
model = joblib.load('titanic_gnb_model.pkl')

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")

st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details to predict if they would have survived the disaster.")

# 2. User Inputs
col1, col2 = st.columns(2)

with col1:
    p_class = st.selectbox("Passenger Class", [1, 2, 3], help="1 = First, 2 = Second, 3 = Third")
    sex_label = st.selectbox("Sex", ["Male", "Female"])
    
with col2:
    age = st.number_input("Age", min_value=0, max_value=100, value=30)
    fare = st.number_input("Fare (Ticket Price)", min_value=0.0, value=32.0)

# 3. Preprocessing (Matching your notebook's logic)
# Map Sex to numeric: male -> 0, female -> 1
sex = 0 if sex_label == "Male" else 1

# Create a DataFrame for prediction with exact column names from training
input_df = pd.DataFrame({
    'p_class': [p_class],
    'sex': [sex],
    'age': [age],
    'fare': [fare]
})

# 4. Prediction
if st.button("Predict Survival"):
    # Perform prediction
    prediction = model.predict(input_df)
    
    st.divider()
    if prediction[0] == 1:
        st.success("✨ The passenger is predicted to **SURVIVE**.")
    else:
        st.error("💀 The passenger is predicted **NOT TO SURVIVE**.")

st.info("Note: This prediction is based on a Gaussian Naive Bayes model.")