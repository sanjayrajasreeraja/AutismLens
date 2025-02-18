import os
import datetime
import io
import numpy as np
import streamlit as st
import tensorflow as tf
import gc
import joblib
import cv2
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import pandas as pd
import requests
from groq import Groq

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

##########################
# 1. Load Models         #
##########################

@st.cache_resource
def load_face_mri_model():
    return load_model("models/model.keras", compile=False)

@st.cache_resource
def load_meta_model():
    return joblib.load("models/MetaModel.pkl")

@st.cache_resource
def load_asq_model():
    return joblib.load("models/ASQmodel.pkl")

face_mri_model = load_face_mri_model()
meta_model = load_meta_model()
asq_model = load_asq_model()

##########################
# 2. Constants & Classes #
##########################

class_names = ['Autistic', 'Non_Autistic', 'autism', 'normal']  # Your class names

questions = {
    "A1": "Does your child respond when their name is called?",
    "A2": "Does your child maintain eye contact during interactions?",
    "A3": "Does your child point or gesture to share interests or requests?",
    "A4": "Does your child engage in pretend or make-believe play?",
    "A5": "Does your child follow your gaze or point to look at something?",
    "A6": "Does your child show interest in other children (e.g., playing together)?",
    "A7": "Does your child use simple gestures like waving or nodding?",
    "A8": "Does your child have any unusual sensitivity to sensory experiences (sound, touch)?",
    "A9": "Does your child repeat certain behaviors or routines consistently?",
    "A10": "Is your child easily comforted when upset?"
}

##########################
# 3. Utility Functions   #
##########################

def reset_session():
    K.clear_session()
    gc.collect()


def preprocess_image_bytes(file_bytes, target_size=(224, 224)):
    """
    Preprocess an uploaded image directly from its bytes.
    """
    try:
        # Read the file bytes as a PIL Image
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        # Resize and convert to array
        img = img.resize(target_size)
        img_array = img_to_array(img) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        st.stop()

##########################
# 4. Groq LLM            #
##########################
# Create Groq client with your API key (set as an environment variable: GROQ_API_KEY)
client = Groq(
    api_key="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",
)

def generate_autism_report_text(face_confidence, mri_confidence, asq_summary, meta_confidence):
    """
    Uses the Groq chat completion API to generate the autism report text.
    """
    prompt = f"""
    You are a clinical assistant generating an autism assessment report based on AI analysis.

    1. **Facial Image Analysis**:
       - AI Model Confidence (Autistic): {face_confidence:.2f}%
    2. **MRI Scan Analysis**:
       - AI Model Confidence (Autistic): {mri_confidence:.2f}%
    3. **ASQ Model Confidence**:
       {asq_summary}
    4. **Meta-Model Confidence**:
       - Probability of Autism from Meta-Model: {meta_confidence:.2f}%

    Based on this data, write a professional autism assessment report in a clear and structured format for doctors and parents. Also, in detail include steps to take and preventions that can be done.
    """

    # Make the Groq chat completion request
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt},
        ],
        model="llama-3.3-70b-versatile",  # example model
    )
    return chat_completion.choices[0].message.content


##########################
# 5. PDF Generation      #
##########################

def generate_pdf_report(
    patient_name, age_in_months, gender,
    face_confidence, mri_confidence, asq_confidence,
    meta_confidence,
    asq_summary, llm_report,
    face_image_bytes=None, mri_image_bytes=None
):
    pdf_filename = f"{patient_name.replace(' ', '_')}_Autism_Report.pdf"
    c = canvas.Canvas(pdf_filename, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(200, height - 50, "Autism Detection Report")

    # Patient Information
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 80, f"Patient Name: {patient_name}")
    c.drawString(50, height - 100, f"Age (months): {age_in_months}")
    c.drawString(50, height - 120, f"Gender: {gender}")
    c.drawString(50, height - 140, f"Date of Assessment: {datetime.date.today()}")

    y_offset = height - 180

    # Face Image
    if face_image_bytes:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y_offset, "Facial Image:")
        y_offset -= 15
        face_img_reader = ImageReader(io.BytesIO(face_image_bytes))
        c.drawImage(face_img_reader, 50, y_offset - 200, width=200, height=200)
        y_offset -= 220

    # MRI Image
    if mri_image_bytes:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(300, height - 180, "MRI Image:")
        mri_img_reader = ImageReader(io.BytesIO(mri_image_bytes))
        c.drawImage(mri_img_reader, 300, height - 200, width=200, height=200)

    # AI Predictions
    y_position = y_offset - 40
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_position, "AI Model Predictions")
    y_position -= 20
    c.setFont("Helvetica", 12)
    c.drawString(50, y_position, f"Facial Image Confidence (Autistic): {face_confidence:.2f}%")
    y_position -= 20
    c.drawString(50, y_position, f"MRI Confidence (Autistic): {mri_confidence:.2f}%")
    y_position -= 20
    c.drawString(50, y_position, f"ASQ Model Confidence (Autistic): {asq_confidence:.2f}%")
    y_position -= 20
    c.drawString(50, y_position, f"Meta-Model Confidence (Autistic): {meta_confidence:.2f}%")

    # LLM-Generated Diagnosis Summary
    y_position -= 60
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_position, "AI-Generated Report:")
    y_position -= 20
    c.setFont("Helvetica", 10)
    text = c.beginText(50, y_position)
    text.setLeading(14)
    for line in llm_report.split("\n"):
        text.textLine(line)
    c.drawText(text)

    c.save()
    return pdf_filename

##########################
# 6. Streamlit App       #
##########################

st.title("🧠 AutismLens: AI-Powered Autism Detection (Groq LLM, No Local Save)")

# Patient Info
patient_name = st.text_input("Patient Name", "John Doe")
age_in_months = st.number_input("Age in months", min_value=1, max_value=240, value=36)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])

# Upload Images
st.subheader("Upload Images")
face_image = st.file_uploader("Facial Image", type=["jpg", "jpeg", "png"])
mri_image = st.file_uploader("MRI Scan", type=["jpg", "jpeg", "png"])

# Display questions
st.subheader("Autism Screening Questionnaire (ASQ)")
asq_features_dict = {}
for q_key, question_text in questions.items():
    response = st.radio(question_text, ["Yes", "No"], key=q_key)
    asq_features_dict[q_key] = 1 if response == "Yes" else 0

st.text("\n")

if st.button("🔍 Submit and Predict"):
    if face_image is None or mri_image is None:
        st.error("Please upload both face and MRI images.")
        st.stop()

    reset_session()

    face_mri_model = load_face_mri_model()
    meta_model = load_meta_model()
    asq_model = load_asq_model()

    # Convert the uploaded files to bytes
    face_bytes = face_image.getvalue()
    mri_bytes = mri_image.getvalue()

    # Preprocess in memory
    face_input = preprocess_image_bytes(face_bytes)
    mri_input = preprocess_image_bytes(mri_bytes)

    # Face Predictions
    face_predictions = face_mri_model.predict(face_input)
    face_prediction_probabilities = face_predictions[0] * 100
    # Index for 'Autistic'
    face_confidence = face_prediction_probabilities[class_names.index("Autistic")]

    # MRI Predictions
    mri_predictions = face_mri_model.predict(mri_input)
    mri_prediction_probabilities = mri_predictions[0] * 100
    # Index for 'autism'
    mri_confidence = mri_prediction_probabilities[class_names.index("autism")]

    # ASQ Model Prediction
    asq_features_dict["Age_Mons"] = age_in_months
    asq_features_dict["Sex"] = 1 if gender == "Male" else 0
    asq_df = pd.DataFrame({k: [v] for k, v in asq_features_dict.items()})
    asq_prediction = asq_model.predict_proba(asq_df)[0][1] * 100

    # Meta-Model Prediction
    meta_input = [[face_confidence, mri_confidence, asq_prediction]]
    meta_probabilities = meta_model.predict_proba(meta_input)[0]
    prob_autistic = meta_probabilities[1] * 100

    # Summaries
    yes_count = sum(value for key, value in asq_features_dict.items() if key.startswith("A"))
    no_count = 10 - yes_count
    asq_summary = f"ASQ Model Confidence: {asq_prediction:.2f}%\nYes={yes_count}, No={no_count} out of 10."

    # LLM Report (Groq)
    llm_report = generate_autism_report_text(face_confidence, mri_confidence, asq_summary, prob_autistic)

    pdf_filename = generate_pdf_report(
        patient_name,
        age_in_months,
        gender,
        face_confidence,
        mri_confidence,
        asq_prediction,
        prob_autistic,
        asq_summary,
        llm_report,
        face_image_bytes=face_bytes,
        mri_image_bytes=mri_bytes
    )

    st.header("📈 Prediction Results")
    st.markdown(f"- **Face Model (Autistic)**: {face_confidence:.2f}%")
    st.markdown(f"- **MRI Model (Autistic)**: {mri_confidence:.2f}%")
    st.markdown(f"- **ASQ Model (Autistic)**: {asq_prediction:.2f}%")
    st.markdown(f"- **Meta-Model (Autistic)**: {prob_autistic:.2f}%")

    if prob_autistic > 60:
        st.warning("High likelihood of Autism. Please consult a specialist.")
    else:
        st.success("Low likelihood of Autism.")

    with open(pdf_filename, "rb") as pdf_file:
        pdf_bytes = pdf_file.read()
    st.download_button("📥 Download Report", data=pdf_bytes, file_name=pdf_filename, mime="application/pdf")
