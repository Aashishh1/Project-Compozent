import streamlit as st
import numpy as np
import string
import pickle
 
model = pickle.load(open('model_pkl.pkl','rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))


def main():
    st.sidebar.header("Diabetes Risk Prediction for Females")
    st.sidebar.text("This is a web app that predicts whether a female is at risk for Diabetes.")
    st.sidebar.text("The model used: AdaBoost Classifier")

    st.header("Enter your health information below:")
    Pregnancies = st.slider("Number of Pregnancies:", 0, 16, step=1)
    Glucose = st.slider("Glucose Level:", 74, 200, step=1)
    BloodPressure = st.slider("Blood Pressure (mm Hg):", 30, 130, step=1)
    SkinThickness = st.slider("Skin Thickness (mm):", 0, 100, step=1)
    Insulin = st.slider("Insulin Level (µU/mL):", 0, 200, step=1)
    BMI = st.slider("BMI (Body Mass Index):", 14.0, 60.0, step=0.1)
    DiabetesPedigreeFunction = st.slider("Diabetes Pedigree Function:", 0.0, 2.5, step=0.01)
    Age = st.slider("Age:", 10, 100, step=1)

    inputs = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

    scaled_features = scaler.transform(inputs)
    print(scaled_features)
    print(scaled_features.shape)

    if st.button('Predict'):
        try:
            result = model.predict(scaled_features)
            print(result)
            if result[0] == 0:
                st.success("Not very probable you will get diabetes soon. Take care of your health!")
            else:
                st.warning("It is probable you might develop diabetes soon. Please consult a healthcare professional.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            
if __name__ == '__main__':
    main()