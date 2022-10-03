import streamlit as st
import pandas as pd
import numpy as np
import joblib
from predict import *

loc_file=open("data/locations.txt", "r", encoding="utf-8")
location_list=loc_file.read().split("\n")

user_input={'Location': None, 
    'Rooms': None, 
    'Meters^2':None, 
    'Floor': None, 
    'House Type': None}

st.title('House Price Prediction')
st.write("Fill in all boxes and press Submit button to get a predicted price.")

with st.form("my_form"):
    user_input['Meters^2']=st.number_input('Square Meters', min_value=1, value=1, format='%i')
    user_input['Rooms']=st.number_input('No. of Rooms', min_value=1, value=1, format='%i')
    user_input['Floor']=st.number_input('Floor', min_value=1, value=1, format='%i')
    user_input['House Type']=st.selectbox('House Type', ['old-build', 'new-build'])
    user_input['Location']=st.selectbox('Location',location_list)

    submitted = st.form_submit_button("Submit")

    if submitted:
        price=price_predict(user_input, 'models/xgb_model.sav', 'models/test_prep_pipe.bin')
        st.success(f'The Predicted House Price is: {int(price)} AZN')