import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
#=========================================
st.set_page_config(
    page_title="Boston House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

#=========================================

@st.cache_resource
def load_model():
    model = joblib.load('predict.joblib')
    
    return model

model = load_model()
#=========================================

#=========================================
st.markdown(
    """
    <style>
        .main-title {
            font-size: 3rem !important;
            font-weight: bold;
            color: #ff6f00;
            text-align: center;
            animation: fadeInDown 1s ease-in-out;
        }
        @keyframes fadeInDown {
            0% { opacity: 0; transform: translateY(-20px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        .prediction-result {
            font-size: 1.5rem;
            color: #1565c0;
            font-weight: bold;
            text-align: center;
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
    </style>
    """,
    unsafe_allow_html=True
)
#=========================================
st.markdown("<div class='main-title'>Boston House Price Predictor 🏠</div>", unsafe_allow_html=True)
st.write(
    """
    This app predicts the **median value of owner-occupied homes** in Boston suburbs.
    Enter the details of a neighborhood, and the model will estimate the house price.
    This is based on a Linear Regression model trained on the classic Boston Housing dataset.
    """
)
st.markdown("---")
#=========================================
st.sidebar.header("Input Features")

#a function that act as the input collector
#in this case it collects data and use pandas to covert it to a dataframe
def user_input_features():
    Year = st.sidebar.slider('enter YEAR', 2000,2030)
    NA_Sales = st.sidebar.slider('enter NA_Sales', 0.0,100.0,10.0)
    EU_Sales = st.sidebar.slider('enter EU_sales', 0.0,100.0,10.0)
    JP_Sales = st.sidebar.slider('enter JP_Sales', 0.0,100.0,10.0)
    Other_Sales = st.sidebar.slider('enter Other_sales', 0.0,100.0,10.0)

    data = {
        'Year':  Year,
        'NA_Sales': NA_Sales,
        'EU_Sales': EU_Sales ,
        'JP_Sales': JP_Sales, 
        'Other_Sales': Other_Sales,   
    }
        
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()
#=========================================
# --- Main Panel ---
st.header("Your Input")
st.dataframe(input_df)
#=========================================
if st.sidebar.button("Predict global_sales"): # meaning if i click the sidebar button
    with st.spinner("Calculating price prediction..."): # show a spiner with this message
        time.sleep(1.5)  # Simulate processing delay
        # scaled_input = scaler.transform(input_df) #scale the user input
        prediction = model.predict(input_df)
        predicted_price = prediction[0]
    
    st.markdown(f"<div class='prediction-result'>Predicted global_sales: {predicted_price:,.2f}</div>", unsafe_allow_html=True)
    st.balloons()

st.markdown("---")
st.write("Disclaimer: The Boston Housing dataset has known ethical issues. This app is for educational purposes only.")

#=========================================
#=========================================
#=========================================
