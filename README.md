# AutismLens

## Overview
AutismLens is a comprehensive AI-driven diagnostic tool that leverages deep learning and machine learning techniques to assist in the early detection of Autism Spectrum Disorder (ASD). This project integrates multiple models to analyze facial images, MRI scans, and Autism Spectrum Quotient (ASQ) survey responses, culminating in a powerful meta-model that enhances diagnostic accuracy. The system is deployed as a user-friendly Streamlit web application, providing accessible diagnostics and automated report generation using Large Language Models (LLM).

## Features
- **Image Classification Model (MobileNet)**: Classifies facial and MRI images into autistic and non-autistic categories.
- **ASQ-based Autism Prediction Model**: Uses survey data with attributes like age, sex, and ethnicity to predict ASD likelihood.
- **Meta-Model (Random Forest)**: Aggregates predictions from image and survey models to enhance accuracy.
- **Web Application (Streamlit)**: Enables users to upload images, complete ASQ, and receive a diagnostic report.
- **LLM-Powered Reports (Groq)**: Generates a detailed patient report based on model predictions.

---

## Models and Performance

### 1. **Image Classification Model (MobileNet)**
- **Dataset**:
  - 700 images per class (train) and 500 images per class (test)
  - Classes: `face_autism`, `face_non_autism`, `mri_autism`, `mri_non_autism`
- **Performance Metrics**:
  - Accuracy: **81%**
  - F1 Score (Macro): **0.8088**
  - F1 Score (Micro): **0.791**
  - F1 Score (Weighted): **0.8018**
  - AUC-ROC Scores:
    - Autistic: **0.9521**
    - Non_Autistic: **0.9521**
    - Autism MRI: **0.9571**
    - Normal MRI: **0.9569**

#### Classification Report
| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Autistic Face | 0.84 | 0.64 | 0.73 | 500 |
| Non-Autistic Face | 0.71 | 0.88 | 0.79 | 500 |
| Autism MRI | 0.75 | 0.85 | 0.80 | 500 |
| Normal MRI | 0.82 | 0.72 | 0.77 | 500 |

---

### 2. **ASQ-based Autism Prediction Model (Logistic Regression)**
- **Dataset**:
  - 1100 rows from Kaggle (initially unbalanced)
  - Balanced using SMOTE to **1000 rows (500 autistic, 500 non-autistic)**
- **Performance Metrics**:
  - Accuracy: **99%**
  - Precision, Recall, F1-Score: **0.99 - 1.00**

#### Feature Importance (Logistic Regression)
| Feature | Importance |
|---|---|
| A2 | 2.8358 |
| A9 | 2.7129 |
| A4 | 2.5222 |
| A8 | 2.4021 |
| A7 | 2.3924 |
| A5 | 2.3863 |
| A6 | 2.3457 |
| A10 | 2.0671 |
| A1 | 2.0192 |
| A3 | 1.9457 |
| Sex | 0.2458 |
| Age_Mons | 0.0184 |

---

### 3. **Meta-Model (Random Forest)**
- **Dataset Generation**:
  - Combined test outputs from the image classification and ASQ models
  - Created synthetic dataset with **200,000 samples (100,000 per marker)**
- **Performance Metrics**:
  - **Validation Metrics**:
    - Accuracy: **99%**
    - Precision, Recall, F1-Score: **98.12%**
    - ROC-AUC: **98.9%**
  - **Test Metrics**:
    - Accuracy: **99.1%**
    - Precision, Recall, F1-Score: **98.12%**
    - ROC-AUC: **98.9%**

#### Feature Importance (Random Forest)
| Feature | Importance |
|---|---|
| Probability - Autism Face | 0.36 |
| Probability - Autism MRI | 0.36 |
| Probability - ASQ Autism | 0.28 |

---

## Web Application (Streamlit)
The AutismLens web application provides an intuitive interface for ASD assessment:
1. **Image Upload**: Users can upload facial and MRI images.
2. **ASQ Survey**: Users complete a 10-question Autism Spectrum Quotient survey.
3. **Model Inference**: The system runs each input through its respective model.
4. **Meta-Model Prediction**: Results from all models are aggregated for a final diagnostic.
5. **Report Generation**: The Groq-powered LLM generates a comprehensive patient report.
