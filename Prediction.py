import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
from fpdf import FPDF
import tempfile
import datetime
import seaborn as sns
import io

# Function to upload model
def upload_model(model_name):
    uploaded_file = st.file_uploader(f"Upload {model_name} Model", type="joblib")
    if uploaded_file is not None:
        model = joblib.load(uploaded_file)
        st.success(f"{model_name} Model uploaded successfully!")
        return model
    else:
        st.warning(f"Please upload the {model_name} model to proceed.")
        return None

# Sidebar for navigation
st.sidebar.title("Available Prediction Models")
tabs = st.sidebar.radio("Choose a Prediction Task", ["Lung Cancer Prediction", "Ozone Level Prediction"])


# Function to create PDF
def create_pdf(location, name, input_data, prediction_result, prediction_type="lung_cancer", prediction_label=None, visualization=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    today_date = datetime.date.today()

    if prediction_type == "lung_cancer":
        pdf.cell(100, 10, f"Name: {name}", ln=False)
        pdf.cell(100, 10, f"Report Date: {today_date}", ln=True)
    elif prediction_type == "ozone_level":
        pdf.cell(100, 10, f"Location: {location}", ln=False)
        pdf.cell(100, 10, f"Report Date: {today_date}", ln=True)

    # Input data table
    pdf.cell(200, 10, "Input Data:", ln=True)
    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(95, 10, "Feature", border=1)
    pdf.cell(95, 10, "Value", border=1, ln=True)
    pdf.set_font("Arial", size=12)

    for key, value in input_data.items():
        pdf.cell(95, 10, key, border=1)
        pdf.cell(95, 10, str(value), border=1, ln=True)

    pdf.cell(200, 10, ln=True)  # Add a blank line
    pdf.set_font("Arial", style='B', size=12)

    if prediction_type == "lung_cancer":
        pdf.cell(200, 10, f"Prediction Result: {prediction_label}", ln=True)
    elif prediction_type == "ozone_level":
        pdf.cell(200, 10, f"Predicted Ozone Level: {prediction_result:.2f}", ln=True)

    pdf_output = "output.pdf"
    pdf.output(pdf_output)
    return pdf_output

# Guidance for Lung Cancer
def display_guidance_lung_cancer():
    st.subheader("Lung Cancer Prediction Input Guidance")
    st.write(""" 
    - **AGE**: Age of the patient (years).
    - **SMOKING**: Smoking habit (1: No, 2: Yes).
    - **YELLOW FINGERS**: Presence of yellow fingers (1: No, 2: Yes).
    - **ANXIETY**: Experience of anxiety (1: No, 2: Yes).
    - **PEER PRESSURE**: Peer pressure (1: No, 2: Yes).
    - **CHRONIC DISEASE**: Presence of chronic diseases (1: No, 2: Yes).
    - **FATIGUE**: Experience of fatigue (1: No, 2: Yes).
    - **WHEEZING**: Wheezing (1: No, 2: Yes).
    - **COUGHING**: Persistent coughing (1: No, 2: Yes).
    - **CHEST PAIN**: Experience of chest pain (1: No, 2: Yes).
    - **ALCOHOL CONSUMING**: Alcohol consumption habit (1: No, 2: Yes).
    - **SHORTNESS OF BREATH**: Experience of shortness of breath (1: No, 2: Yes).
    - **SWALLOWING DIFFICULTY**: Difficulty swallowing (1: No, 2: Yes).
    - **ALLERGY**: Presence of allergies (1: No, 2: Yes).
    """)


# Guidance for Ozone Level
def display_guidance_ozone_level():
    st.subheader("Ozone Level Prediction Input Guidance")
    st.write(""" 
    - **tmpd**: Temperature in degrees Celsius.
    - **dptp**: Dew point temperature in degrees Celsius.
    - **pm25tmean2**: Average PM2.5 concentration (µg/m³).
    - **pm10tmean2**: Average PM10 concentration (µg/m³).
    - **no2tmean2**: Average NO2 concentration (µg/m³).
    """)

# Lung Cancer Prediction Tab
if tabs == "Lung Cancer Prediction":
    st.title("Lung Cancer Prediction")

    lung_cancer_model = upload_model("Lung Cancer")

    if lung_cancer_model:
        # Guidance Dropdown
        if st.radio("Show Input Guidance?", ["No", "Yes"]) == "Yes":
            display_guidance_lung_cancer()

        with st.form(key='lung_cancer_form'):
            features = {
                "AGE": st.number_input('Age', min_value=1, max_value=120),
                "SMOKING": st.selectbox('Smoking (1: No, 2: Yes)', [1, 2]),
                "YELLOW_FINGERS": st.selectbox('Yellow Fingers (1: No, 2: Yes)', [1, 2]),
                "ANXIETY": st.selectbox('Anxiety (1: No, 2: Yes)', [1, 2]),
                "PEER_PRESSURE": st.selectbox('Peer Pressure (1: No, 2: Yes)', [1, 2]),
                "CHRONIC_DISEASE": st.selectbox('Chronic Disease (1: No, 2: Yes)', [1, 2]),
                "FATIGUE": st.selectbox('Fatigue (1: No, 2: Yes)', [1, 2]),
                "WHEEZING": st.selectbox('Wheezing (1: No, 2: Yes)', [1, 2]),
                "COUGHING": st.selectbox('Coughing (1: No, 2: Yes)', [1, 2]),
                "CHEST_PAIN": st.selectbox('Chest Pain (1: No, 2: Yes)', [1, 2]),
                "ALCOHOL_CONSUMING": st.selectbox('Alcohol Consuming (1: No, 2: Yes)', [1, 2]),
                "SHORTNESS_OF_BREATH": st.selectbox('Shortness of Breath (1: No, 2: Yes)', [1, 2]),
                "SWALLOWING_DIFFICULTY": st.selectbox('Swallowing Difficulty (1: No, 2: Yes)', [1, 2]),
                "ALLERGY": st.selectbox('Allergy (1: No, 2: Yes)', [1, 2]),
            }

            submit_button = st.form_submit_button("Predict")

        if submit_button:
            feature_values = np.array([list(features.values())])
            prediction = lung_cancer_model.predict(feature_values)
            prediction_label = "Lung Cancer Present" if prediction[0] == 1 else "No Lung Cancer"
            st.write(f"Prediction: **{prediction_label}**")

            pdf_path = create_pdf(
                None, "Lung Cancer Prediction", features, prediction[0], prediction_type="lung_cancer", prediction_label=prediction_label
            )
            with open(pdf_path, "rb") as pdf_file:
                st.download_button("Download Report as PDF", pdf_file, file_name="Lung_Cancer_Prediction.pdf")

# Ozone Level Prediction Tab
elif tabs == "Ozone Level Prediction":
    st.title("Ozone Level Prediction")

    ozone_model = upload_model("Ozone Level")

    if ozone_model:
        # Guidance Dropdown
        if st.radio("Show Input Guidance?", ["No", "Yes"]) == "Yes":
            display_guidance_ozone_level()

        with st.form(key='ozone_form'):
            features = {
                "tmpd": st.number_input("Temperature (°C)"),
                "dptp": st.number_input("Dew Point Temperature (°C)"),
                "pm25tmean2": st.number_input("PM2.5 Concentration"),
                "pm10tmean2": st.number_input("PM10 Concentration"),
                "no2tmean2": st.number_input("NO2 Concentration"),
            }

            submit_button = st.form_submit_button("Predict")

        if submit_button:
            feature_values = np.array([list(features.values())])
            prediction = ozone_model.predict(feature_values)[0]
            st.write(f"Predicted Ozone Level: **{prediction:.2f} µg/m³**")

            pdf_path = create_pdf(
                None, "Ozone Level Prediction", features, prediction, prediction_type="ozone_level"
            )
            with open(pdf_path, "rb") as pdf_file:
                st.download_button("Download Report as PDF", pdf_file, file_name="Ozone_Level_Prediction.pdf")
