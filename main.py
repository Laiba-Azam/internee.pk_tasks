import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler

df=pd.read_csv("Breast_cancer_data.csv")
st.title("Breast Detection ML model")
st.markdown("The application uses ML algorithm to detect breast tumor based on the input provided with upto 94% accuracy")
radius=st.text_input('Mean Radius',value='0')
texture=st.text_input('Mean Texture',value='0')
smoothness=st.text_input('Mean Smoothness',value='0')
if not radius:
    radius=0
if not texture:
    texture=0
if not smoothness:
    smoothness=0
radius=float(radius)
texture=float(texture)
smoothness=float(smoothness)



# Load the model from the file
with open('model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)
input_data=pd.DataFrame({
    'mean_radius':[radius],
    'mean_texture':[texture],
    'mean_smoothness':[smoothness]
})
normalizer=MinMaxScaler()
normalize=normalizer.fit_transform(input_data)

if st.button('Predict'):
    prediction = loaded_model.predict_proba(normalize)
    predicted_class = prediction.argmax(axis=1)
    if predicted_class[0] == 0:
        st.success('No breast tumor detected. Probability: {}'.format(prediction[0, 0]))
    elif predicted_class[0]==1:
        st.error('Breast tumor detected. Probability: {}'.format(prediction[0, 1]))

