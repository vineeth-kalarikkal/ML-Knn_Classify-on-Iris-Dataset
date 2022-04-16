# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 14:43:38 2022

@author: vinee
"""

import streamlit as st
import pandas as pd
import joblib
from PIL import Image


# Loading trained Knn model
model = open("Knn_Classify (2).pkl", "rb")
knn_clf=joblib.load(model)

st.title("Iris flower species Classification App")                    
st.sidebar.title("Features")
# For Loading Images
setosa = Image.open("Setosa.jpg")
versicolor = Image.open("Versicolor.jpg")
virginica = Image.open("Virginica.jpg")                   

parameter_list = ['Sepal length (cm)', 'Sepal Width (cm)', 'Petal Length (cm)', 'Petal Width (cm)']
parameter_input_values=[]
parameter_default_values=['5.2', '3.2', '4.2', '1.2']
values=[]

# Display
for parameter, parameter_df in zip(parameter_list, parameter_default_values):
    values= st.sidebar.slider(label=parameter, key=parameter, value = float(parameter_df), min_value =0.0, max_value = 8.0, step = 0.1)
    parameter_input_values.append(values)
input_variables=pd.DataFrame([parameter_input_values], columns=parameter_list, dtype=float)
st.write(input_variables)

prediction = 0
if st.button("Click Here to Classify"):
    prediction = knn_clf.predict(input_variables)
if prediction == 0:
    st.image(setosa)
elif prediction == 1:
    st.image(versicolor)
else:
    st.image(virginica)

    