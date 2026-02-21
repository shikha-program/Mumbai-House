# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Mumbai House Price Prediction", page_icon="🏠")
st.title("🏠 Mumbai House Price Prediction App")
st.write("Enter property details below:")

# -------------------------------------------------
# STEP 1: Load Pre-trained Model & Encoders
# -------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    encoders = joblib.load("encoder.pkl")
    return model, encoders

model, encoders = load_model()

# -------------------------------------------------
# STEP 2: USER INPUTS
# -------------------------------------------------
age = st.number_input("Age of Property", min_value=0, max_value=100)
city = st.selectbox("City", encoders["City"].classes_)
area = st.selectbox("Area", encoders["Area"].classes_)
property_type = st.selectbox("Property Type", encoders["property_type"].classes_)
years_of_exp = st.number_input("Years of Experience", min_value=0, max_value=50)

# -------------------------------------------------
# STEP 3: PREDICTION
# -------------------------------------------------
if st.button("Predict Price"):
    df = pd.DataFrame({
        "Age": [age],
        "City": [city],
        "Area": [area],
        "property_type": [property_type],
        "Years of Experience": [years_of_exp]
    })
    
    # Encode input using saved encoders
    for col in ["City", "Area", "property_type"]:
        df[col] = encoders[col].transform(df[col])
    
    prediction = model.predict(df)
    st.success(f"💰 Predicted House Price: ₹ {prediction[0]:,.2f}")
