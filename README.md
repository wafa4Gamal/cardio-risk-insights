# 🫀 Cardiovascular Risk — Clinical Insights & Machine Learning Platform

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B?logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Latest-F7931E?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-008000)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DNN-FF6F00?logo=tensorflow)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

An interactive, AI-powered clinical decision support platform designed for Cardiovascular Disease (CVD) risk assessment. The application combines an ensemble of Machine Learning and Deep Learning models, providing clinicians with interpretable predictions through SHAP values, patient clustering, and confidence interval estimations.

---

## 📌 Table of Contents
- [Key Features](#-key-features)
- [Dataset Overview](#-dataset-overview)
- [Model Architecture](#-model-architecture)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [How to Use](#-how-to-use)
- [Tech Stack](#-tech-stack)
- [Contributing](#-contributing)
- [License](#-license)
- [Medical Disclaimer](#-medical-disclaimer)

---

## 🚀 Key Features

1. **Single Patient Assessment:**
   * Input individual patient clinical metrics to generate a **Cardiovascular Risk Score** with 95% Confidence Intervals ($CI$).
   * Interactive **SHAP Force Plots** to provide explainable AI insights into feature contributions for individual predictions.
   * Nearest Neighbors retrieval ($k$-NN) to match and present similar historical patient profiles.

2. **Batch Processing & Export:**
   * Upload CSV files containing bulk clinical records to run parallel predictions and export results directly.

3. **Model Evaluation & Diagnostics:**
   * Performance metrics evaluation (Accuracy, ROC-AUC) across multiple models on holdout test data.
   * Model reliability visualization using probability calibration curves.

4. **Patient Clustering & Profiling:**
   * Patient subgrouping using $K$-Means Clustering to identify risk clusters and behavioral clinical patterns.

---

## 📊 Dataset Overview

This project uses the [Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/mdshamimrahman/cardio-data-set) available on Kaggle.

### Feature Description:

| Feature | Type | Description |
| :--- | :--- | :--- |
| `age` / `age_years` | Numerical | Age (in days / years) |
| `gender` | Categorical | Gender (1: Female, 2: Male) |
| `height` | Numerical | Height (cm) |
| `weight` | Numerical | Weight (kg) |
| `ap_hi` | Numerical | Systolic blood pressure |
| `ap_lo` | Numerical | Diastolic blood pressure |
| `cholesterol` | Categorical | Cholesterol levels (1: normal, 2: above normal, 3: well above normal) |
| `gluc` | Categorical | Glucose levels (1: normal, 2: above normal, 3: well above normal) |
| `smoke` | Binary | Smoking status (0: No, 1: Yes) |
| `alco` | Binary | Alcohol intake (0: No, 1: Yes) |
| `active` | Binary | Physical activity (0: No, 1: Yes) |
| `cardio` | **Target Variable** | Cardiovascular disease presence (0: Absence, 1: Presence) |

---

## 🛠️ Model Architecture

The framework trains and ensembles multiple predictive algorithms:
* **XGBoost Classifier (`xgb`)**
* **Random Forest (`rf` & `rf_simple`)**
* **Logistic Regression (`lr`)**
* **Gradient Boosting (`gb`)**
* **Voting Classifier (`voting`):** Soft-voting ensemble combining top-performing models.
* **Deep Neural Network (`dnn`):** Multi-layer perceptron built with TensorFlow/Keras.
* **K-Means Clustering (`kmeans`):** Patient subgroup segmentation.

---

## 📁 Project Structure

```text
├── cardio_train.csv         # Semicolon-separated clinical dataset
├── project.py               # Data pipeline, preprocessing, and model training routines
├── app.py                   # Streamlit web interface application
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
