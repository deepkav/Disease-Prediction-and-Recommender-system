import streamlit as st
import numpy as np
import zipfile
import tempfile
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import os
import pandas as pd
#from PIL import Image
import base64

def get_data():
    return pd.read_csv("hospitals.csv")

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('back1.png')


st.header('Health Care Recommender Systems')
st.write()
st.subheader('The place to get about your health status')
#image = Image.open('hospital_plus.jfif')
#st.image(image, caption='Hospital Symbol')
st.write()

Symptoms = st.text_area('Symptoms',"Symptom1, Sympton2, ...")
Symptoms = Symptoms.split(",")
Symptom1 = Symptoms[0]
Symptom2 = Symptoms[1]

severity = ['High', 'Low']
Intensity = st.radio('Intensity',severity)
Intensity = severity.index(Intensity)

Duration = st.number_input('How long')

period = ['All Day', 'Morning', 'Night']
Time = st.selectbox('Time',period)
Time = period.index(Time)

Age = st.number_input('Age')

gender = ['Female', 'Male']
Sex = st.radio('Gender',gender)
Sex = gender.index(Sex)

stream = st.file_uploader('main3.zip', type='zip')
if stream is not None:
    myzipfile = zipfile.ZipFile(stream)
    with tempfile.TemporaryDirectory() as tmp_dir:
        myzipfile.extractall(tmp_dir)
        root_folder = myzipfile.namelist()[0] # e.g. "model.h5py"
        model_dir = os.path.join(tmp_dir, root_folder)
        #st.info(f'trying to load model from tmp dir {model_dir}...')
        model = tf.keras.models.load_model(model_dir)

#Predict button
if st.button('Predict'):
    prediction = model.predict([[3, 3, Intensity, Duration, Time]])
    predict = prediction.tolist()
    #st.markdown(f'{predict},{predict[0][0]}')
    if any(predict[0]) > 0.5:
        df = get_data()
        st.markdown(f'### Disease predictions :')
        if predict[0][0] > 0.5:
            st.markdown('Patient may have onset of Diabetes')
        if predict[0][1] > 0.5:
            st.markdown('Patient may have onset of Hepatitis')
            st.markdown('### The diet for Hepatitis should include the following:')
            st.text("1.Plenty of fruits and vegetables.")
            st.text("2.Whole grains such as oats, brown rice, barley, and quinoa.")
            st.text("3.Lean protein such as fish, skinless chicken, egg whites, and beans.")
            st.text("4.Healthy fats like those in nuts, avocados, and olive oil.")
        if predict[0][2] > 0.5:
            st.markdown('Patient may have onset of Malaria')
            st.markdown("### The diet for malaria should include the following:")
            st.text("1.Eat Nutritious Food.")
            st.text("2.Increase Fluid Intake.")
            st.text("3.Increase Protein Intake.")
            st.text("4.Avoid food high in fat content.")
        st.markdown('Patient needs health care')
        st.header('Here are some list of hospitals👇')
        st.map(df[['latitude','longitude']])
        st.dataframe(df)
    else:
        st.markdown('Patient is Fine')


st.header('Diabetes in Females')

no = st.number_input('Number of times pregnant')
pla = st.number_input('Plasma glucose concentration a 2 hours in an oral glucose tolerance test')
dbp = st.number_input('Diastolic blood pressure (mm Hg)')
thic = st.number_input('Triceps skin fold thickness (mm)')
ins = st.number_input('2-Hour serum insulin (mu U/ml)')
bmi = st.number_input('Body mass index (weight in kg/(height in m)^2)')
dpf = st.number_input('Diabetes pedigree function')
age = st.number_input('Age (years)')

stream = st.file_uploader('preg_dia.zip', type='zip')
if stream is not None:
  myzipfile = zipfile.ZipFile(stream)
  with tempfile.TemporaryDirectory() as tmp_dir:
    myzipfile.extractall(tmp_dir)
    root_folder = myzipfile.namelist()[0] # e.g. "model.h5py"
    model_dir = os.path.join(tmp_dir, root_folder)
    #st.info(f'trying to load model from tmp dir {model_dir}...')
    model = tf.keras.models.load_model(model_dir)

#Predict button
if st.button('Predict2'):
    prediction = model.predict([[no, pla, dbp, thic, ins, bmi, dpf, age]])
    if prediction[0][0] > 0.5:
        df = get_data()
        st.markdown('### Patient have onset of diabetes')
        st.markdown('Patient needs health care ')
        st.header('Here are some list of hospitals👇')
        st.map(df[['latitude','longitude']])
        st.dataframe(df)
    else:
        st.markdown('### Patient is fine 👍')