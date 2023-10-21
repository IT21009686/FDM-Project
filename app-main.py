#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# Load your stroke dataset
stroke_dataset = pd.read_csv("healthcare-dataset-stroke-data.csv")

st.set_page_config(page_title="Health Guard", page_icon="🩺")

# Add CSS for dark theme
st.markdown(
    """
    <style>
    .css-1aumxhk {
        background-color: #1E1E1E; 
    }
    .css-1aoc0fo {
        color: white;  
    }
    .css-14vbfpz {
        color: #FF4B4B; 
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<style>body{background-color: #add8e6;}</style>', unsafe_allow_html=True)


# FUNCTION
def user_report_stroke():
    age = st.sidebar.slider('Age', 3, 88, 33)
    bmi = st.sidebar.slider('BMI', 0.0, 67.0, 20.0)
    gender = st.sidebar.radio('Gender', ('Male', 'Female'))
    hypertension = st.sidebar.radio('Hypertension', ('Yes', 'No'))
    heart_disease = st.sidebar.radio('Heart Disease', ('Yes', 'No'))
    ever_married = st.sidebar.radio('Ever Married', ('Yes', 'No'))
    work_type = st.sidebar.selectbox('Work Type', ('Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'))
    Residence_type = st.sidebar.radio('Residence Type', ('Urban', 'Rural'))
    avg_glucose_level = st.sidebar.slider('Average Glucose Level', 55.0, 300.0, 100.0)
    smoking_status = st.sidebar.selectbox('Smoking Status', ('formerly smoked', 'never smoked', 'smokes', 'Unknown'))

    user_report_data = {
        'age': age,
        'bmi': bmi,
        'gender': gender,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': ever_married,
        'work_type': work_type,
        'Residence_type': Residence_type,
        'avg_glucose_level': avg_glucose_level,
        'smoking_status': smoking_status,
    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data


# Define the information about strokes
stroke_info = """
A stroke occurs when there is a sudden interruption in the blood supply to the brain. It can happen to anyone at any time and can lead to severe complications or even death. 

**Why is getting checked out for a stroke important?**
- Quick medical intervention during a stroke can minimize brain damage and potential complications.
- Identifying and managing stroke risk factors can prevent future strokes.
- Timely treatment can improve the chances of recovery and reduce disability.

If you suspect someone is having a stroke, it is crucial to seek immediate medical attention. Remember the FAST acronym:
- **F (Face):** Ask the person to smile. Does one side of the face droop?
- **A (Arms):** Ask the person to raise both arms. Does one arm drift downward?
- **S (Speech):** Ask the person to repeat a simple phrase. Is their speech slurred or strange?
- **T (Time):** If you observe any of these signs, call emergency services immediately. Time is critical during a stroke.

"""

# Streamlit app
st.title('Stroke Checkup')
st.sidebar.header('Navigation')
tabs = st.sidebar.radio("Go to", ('Learn about Stroke', 'Have a checkup'))

if tabs == 'Learn about Stroke':
    st.subheader('Learn about Stroke')
    # Display information about strokes
    st.markdown(stroke_info, unsafe_allow_html=True)

elif tabs == 'Have a checkup':
    # Data Preprocessing
    stroke_dataset = stroke_dataset.drop_duplicates()
    #stroke_dataset.drop(['id'],axis=1,inplace = True)
    stroke_dataset.dropna(inplace=True)
    stroke_dataset = stroke_dataset[stroke_dataset['gender'] != 'Other']
    stroke_dataset.reset_index(drop=True, inplace=True)

    #Define mapping
    gender_mapping = {'Male': 0, 'Female': 1}
    stroke_dataset['gender'] = stroke_dataset['gender'].map(gender_mapping)

    hypertension_mapping = {'No': 0, 'Yes': 1}
    stroke_dataset['hypertension'] = stroke_dataset['hypertension'].map(hypertension_mapping)

    heartdisease_mapping = {'No': 0, 'Yes': 1}
    stroke_dataset['heart_disease'] = stroke_dataset['heart_disease'].map(heartdisease_mapping)

    evermarried_mapping = {'No': 0, 'Yes': 1}
    stroke_dataset['ever_married'] = stroke_dataset['ever_married'].map(evermarried_mapping)

    worktype_mapping = {'Private': 0, 'Self-employed': 1, 'Govt_job' :2, 'children': 3, 'Never_worked':4}
    stroke_dataset['work_type'] = stroke_dataset['work_type'].map(worktype_mapping)

    residencetype_mapping = {'Urban': 0, 'Rural': 1}
    stroke_dataset['Residence_type'] = stroke_dataset['Residence_type'].map(residencetype_mapping)

    smoking_status_mapping = {'formerly smoked': 0, 'never smoked': 1, 'smokes': 2, 'Unknown': 3}
    stroke_dataset['smoking_status'] = stroke_dataset['smoking_status'].map(smoking_status_mapping)


    # Horizontal line function
    def horizontal_line():
        st.markdown('<hr style="border-top: 4px solid #ff4b4b;">', unsafe_allow_html=True)

    # HEADINGS
    st.sidebar.header('Patient Data')
    horizontal_line()
    st.subheader('Training Data Stats')
    st.write(stroke_dataset.describe())

    # X AND Y DATA
    X_stroke = stroke_dataset.drop('stroke', axis='columns')
    y_stroke = stroke_dataset['stroke']

    # Convert categorical variables to numerical using one-hot encoding
    X_encoded_stroke = pd.get_dummies(X_stroke, columns=['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'], drop_first=True)

    # Ensure that the columns in the user report data match those used during model training
    user_data_stroke = user_report_stroke()
    user_data_stroke = user_data_stroke.reindex(columns=X_encoded_stroke.columns, fill_value=0)

    X_train_stroke, X_test_stroke, y_train_stroke, y_test_stroke = train_test_split(X_encoded_stroke, y_stroke, test_size=0.2, random_state=5)

    # PATIENT DATA
    horizontal_line()
    st.subheader('Have a checkup')
    st.write(user_data_stroke)

    # MODEL SELECTION
    model_stroke = RandomForestClassifier()

    # MODEL TRAINING
    model_stroke.fit(X_train_stroke, y_train_stroke)
    user_result_stroke = model_stroke.predict(user_data_stroke)
    
    # VISUALIZATIONS
    horizontal_line()
    st.subheader('Visualised Patient Report - Stroke')

    # Add more visualizations based on your stroke dataset features here.

    if st.button("Predict"):
        # OUTPUT
        st.write('Your Result: ')
        output_stroke = 'Stroke Detected, Contact your Doctor!' if user_result_stroke[0] == 1 else 'Congratulation, No Stroke Detected'
        st.markdown(f'<p style="color: #ff4b4b; font-size: 24px;">{output_stroke}</p>', unsafe_allow_html=True)
        accuracy_percentage_stroke = str(accuracy_score(y_test_stroke, model_stroke.predict(X_test_stroke)) * 100)
        st.write('Accuracy :   ', f'<span style="color: white;">{accuracy_percentage_stroke}%</span>', unsafe_allow_html=True)


