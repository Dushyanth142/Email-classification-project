#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np # ADDED THIS IMPORT
import streamlit as st
from joblib import load # Removed 'dump' as it's not used here
from sklearn.preprocessing import StandardScaler # Keep this if you want StandardScaler class definition

# Make sure you have scaler.pkl and model.pkl uploaded to your Space's root
@st.cache_resource
def load_artifacts(): # Renamed to load both model and scaler
    model = load('src/model.pkl')
    scaler = load('src/scaler.pkl')
    return model, scaler

# This function now takes the pre-loaded scaler object
def Scaling(x_pred, scaler_obj): # Changed parameter name for clarity
    scaled = scaler_obj.transform(x_pred) # Use scaler_obj
    return scaled

@st.cache_data
def Make_predictions(_model_obj, x_pred): # Pass the loaded model object
    pred = _model_obj.predict(x_pred) # Use model_obj
    bool_predictions = (pred == 1)
    string_predictions = np.where(bool_predictions, 'Open_emails', 'Not_open')
    predictions_series = pd.Series(string_predictions, name='predictions') # Renamed for clarity
    return predictions_series

@st.cache_data
def download(prediction_series): # Accepts the Series
    csv_data = prediction_series.to_csv(index=False) # Convert Series to CSV string
    st.download_button(
        label="Download predictions as CSV",
        data=csv_data, # Pass the CSV string
        file_name="predictions.csv",
        mime="text/csv",
    )

# Function to reset inputs (ensure state keys match your widgets)
def clear_inputs():
    st.session_state.Customer_Age = 0
    st.session_state.Emails_Opened = 0
    st.session_state.Emails_Clicked = 0
    st.session_state.Purchase_History = 0.0 # Use 0.0 for float inputs
    st.session_state.Time_Spent_On_Website = 0.0 # Use 0.0 for float inputs
    st.session_state.Days_Since_Last_Open = 0
    st.session_state.Customer_Engagement_Score = 0.0 # Use 0.0 for float inputs
    st.session_state.Clicked_Previous_Emails = 'Not_Clicked'
    st.session_state.Device_Type = 'Desktop'
    st.session_state.uploaded_file = None


if __name__ == "__main__":
    st.title("Email Classification")

    # Load model and scaler once at app startup
    model, scaler = load_artifacts()

    # Initialize X for batch prediction outside sidebar to control scope
    X_batch = None # Renamed to avoid confusion with X in input fields

    with st.sidebar:
        st.header('For batch predictions, upload a file here')
        # Removed: import streamlit as st (already imported at the top)
        uploaded_file = st.file_uploader("Choose a CSV or XLSX file", type=["csv", "xlsx"])

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    X_batch = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(".xlsx"):
                    X_batch = pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"Error reading file: {e}") # Changed to st.error for visibility

    # Input fields for single prediction
    col1,col2,col3 = st.columns(3)
    with col1:
        # Added key for number_input and default values for session_state clear
        Customer_Age=st.number_input('Customer Age',0,120,value=None,step=1,format='%d',key='Customer_Age',placeholder='Enter the age')
        Emails_Opened=st.number_input('Emails_Opened',value=None,step=1,format='%d',key='Emails_Opened',placeholder='Enter an integer')
        Emails_Clicked=st.number_input('Emails_Clicked',value=None,step=1,format='%d',key='Emails_Clicked',placeholder='Enter an integer')
    with col2:
        Purchase_History=st.number_input('Purchase_History',value=None,step=0.1,format='%0.1f',key='Purchase_History',placeholder='Enter amount in Dollars')
        Time_Spent_On_Website=st.number_input('Time_Spent_On_Website',value=None,step=0.1,format='%0.1f',key='Time_Spent_On_Website',placeholder='Enter the time')
        Days_Since_Last_Open=st.number_input('Days_Since_Last_Open',value=None,step=1,format='%d',key='Days_Since_Last_Open',placeholder='Enter an integer')
    with col3:
        Customer_Engagement_Score=st.number_input('Customer_Engagement_Score',value=None,step=0.1,format='%0.1f',key='Customer_Engagement_Score',placeholder='Enter a value')
        Clicked_Previous_Emails=st.selectbox('Clicked_Previous_Emails',['Clicked','Not_Clicked'],index=None,key='Clicked_Previous_Emails')
        Device_Type=st.selectbox('Device_Type',['Desktop','Mobile'],index=None,key='Device_Type')

    st.subheader('Ask the model to predict')

    # Prediction button and logic
    col4,col5=st.columns(2)
    with col4 :
        pred_btn = st.button('predict',type='primary',key='pred_btn')
        if pred_btn:
            if X_batch is not None: # Use X_batch for batch prediction path
                # Apply scaling using the loaded scaler
                x_p_scaled = Scaling(X_batch, scaler)
                # Make predictions
                pred_result_series = Make_predictions(model, x_p_scaled)
                # Download predictions
                download(pred_result_series)
                with st.expander("See full predicted table"):
                    st.write(pred_result_series) # Write the Series
            else:
                # Handle single prediction input
                # Create a list of input values in the correct order for your model
                # Ensure the order matches the features your model was trained on
                input_values = np.array([
                    Customer_Age, Emails_Opened, Emails_Clicked, Purchase_History, Time_Spent_On_Website,
                    Days_Since_Last_Open, Customer_Engagement_Score,
                    1 if Clicked_Previous_Emails=='Clicked' else 0,
                    1 if Device_Type=='Mobile' else 0
                ]).reshape(1, -1) # Reshape for single sample

                # Apply scaling using the loaded scaler
                scaled_input = Scaling(input_values, scaler)

                # Make prediction
                single_pred_series = Make_predictions(model, scaled_input)
                single_pred_string = single_pred_series.iloc[0] # Get the string result

                if single_pred_string == 'Open_emails':
                    st.write('This customer is likely to respond for Advertisements through Emails')
                else:
                    st.write('This customer is not willing to receive Advertisements through Emails')

    with col5:
        clear = st.button('clear', key='clear_btn', on_click=clear_inputs)