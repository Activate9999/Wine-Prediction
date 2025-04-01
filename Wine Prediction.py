import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import urllib.error

st.title('🍷 Wine Quality Prediction App')

st.info('This app builds a machine learning model to predict the quality of wine based on its features!')

# Correct URL for Wine Quality Dataset (red wine dataset)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

# Load the dataset with error handling
try:
    df = pd.read_csv(url, delimiter=';')
except urllib.error.HTTPError as e:
    st.error(f"HTTP Error: {e.code} - {e.reason}")
    st.stop()  # Stop the app if the dataset cannot be loaded
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.stop()  # Stop the app if any other error occurs

with st.expander('Data'):
    st.write('**Raw Data**')
    st.write(df)
    
    st.write('**Features (X)**')
    X_raw = df.drop('quality', axis=1)
    st.write(X_raw)
    
    st.write('**Target (y)**')
    y_raw = df['quality']
    
with st.expander('Data visualization'):
    st.bar_chart(df['quality'].value_counts())

# Input features
with st.sidebar:
    st.header('Input Features')
    fixed_acidity = st.slider('Fixed Acidity', 4.0, 15.0, 7.4)
    volatile_acidity = st.slider('Volatile Acidity', 0.1, 1.5, 0.52)
    citric_acid = st.slider('Citric Acid', 0.0, 1.0, 0.26)
    residual_sugar = st.slider('Residual Sugar', 0.9, 15.0, 2.6)
    chlorides = st.slider('Chlorides', 0.01, 0.1, 0.04)
    free_sulfur_dioxide = st.slider('Free Sulfur Dioxide', 1, 72, 15)
    total_sulfur_dioxide = st.slider('Total Sulfur Dioxide', 6, 289, 46)
    density = st.slider('Density', 0.99, 1.05, 0.998)
    pH = st.slider('pH', 2.5, 4.0, 3.31)
    sulphates = st.slider('Sulphates', 0.3, 2.0, 0.68)
    alcohol = st.slider('Alcohol', 8.0, 15.0, 10.5)
    
    # Create a DataFrame for the input features
    data = {
        'fixed acidity': fixed_acidity,
        'volatile acidity': volatile_acidity,
        'citric acid': citric_acid,
        'residual sugar': residual_sugar,
        'chlorides': chlorides,
        'free sulfur dioxide': free_sulfur_dioxide,
        'total sulfur dioxide': total_sulfur_dioxide,
        'density': density,
        'pH': pH,
        'sulphates': sulphates,
        'alcohol': alcohol
    }
    input_df = pd.DataFrame(data, index=[0])

with st.expander('Input Features'):
    st.write('**Input Wine Features**')
    st.write(input_df)

# Data preparation
# Encode X and y
X = df.drop('quality', axis=1)
y = df['quality']

# Model training and inference
## Train the ML model
clf = RandomForestClassifier()
clf.fit(X, y)

## Apply model to make predictions
prediction = clf.predict(input_df)
prediction_proba = clf.predict_proba(input_df)

df_prediction_proba = pd.DataFrame(prediction_proba)
df_prediction_proba.columns = sorted(y.unique())

# Display predicted wine quality
st.subheader('Predicted Wine Quality')
st.dataframe(df_prediction_proba)

wine_quality = np.array(sorted(y.unique()))
st.success(f"Predicted Wine Quality: {prediction[0]}")