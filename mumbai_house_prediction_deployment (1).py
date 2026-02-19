# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("model.pkl")
encoders = joblib.load("encoder.pkl")

st.set_page_config(page_title="Mumbai House Price Prediction", page_icon="🏠")

st.title("🏠 Mumbai House Price Prediction App")

# Inputs (NO INDENTATION HERE)
age = st.number_input("Age of Property", min_value=0, max_value=100)

city = st.selectbox("City", encoders["City"].classes_)
area = st.selectbox("Area", encoders["Area"].classes_)
property_type = st.selectbox("Property Type", encoders["property_type"].classes_)

years_of_exp = st.number_input("Years of Experience", min_value=0, max_value=50)

if st.button("Predict Price"):

    df = pd.DataFrame({
        "Age": [age],
        "City": [city],
        "Area": [area],
        "property_type": [property_type],
        "Years of Experience": [years_of_exp]
    })

    for col in ["City", "Area", "property_type"]:
        df[col] = encoders[col].transform(df[col])

    prediction = model.predict(df)

    st.success(f"💰 Predicted House Price: ₹ {prediction[0]:,.2f}")
