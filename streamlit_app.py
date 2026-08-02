import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from typing import Tuple, Dict, Any
import os
from pathlib import Path
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
import shap
import tensorflow as tf

from project import (
    load_data,
    eda,
    preprocess_data,
    split_data,
    resample_and_scale,
    clustering_helper,
    deep_learning,
    train_xgb,
    train_rf,
    train_lr,
    train_voting,
    train_kmeans,
    train_gb,
    train_rf_simple,
    ensemble_evaluation,
    get_calibration_curves
)

st.set_page_config(page_title="Cardio Risk — Clinical Insights", layout="wide")

@st.cache_data
def load_and_prepare(path: str):
    df = load_data(path)
    df = eda(df, plot=False)
    df_cleaned = preprocess_data(df)
    X_train, y_train, X_test, y_test = split_data(df_cleaned)
    return df_cleaned, X_train, y_train, X_test, y_test

def load_or_train(model_dir: Path, name: str, train_func):
    path = model_dir / f"{name}.joblib"
    if path.is_file():
        return joblib.load(path)
    model = train_func()
    if name != "dnn":
        joblib.dump(model, path)
    return model

@st.cache_resource
def get_models_cached(X_train_res: pd.DataFrame, y_train_res: pd.Series, X_test_scaled: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    models: Dict[str, Any] = {}
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    models["xgb"] = load_or_train(model_dir, "xgb", lambda: train_xgb(X_train_res, y_train_res))
    models["rf"] = load_or_train(model_dir, "rf", lambda: train_rf(X_train_res, y_train_res))
    models["lr"] = load_or_train(model_dir, "lr", lambda: train_lr(X_train_res, y_train_res))
    models["gb"] = load_or_train(model_dir, "gb", lambda: train_gb(X_train_res, y_train_res))
    models["rf_simple"] = load_or_train(model_dir, "rf_simple", lambda: train_rf_simple(X_train_res, y_train_res))

    voting_path = model_dir / "voting.joblib"
    if voting_path.is_file():
        models["voting"] = joblib.load(voting_path)
    else:
        estimators = [("xgb", models["xgb"]), ("rf", models["rf"]), ("lr", models["lr"])]
        models["voting"] = train_voting(X_train_res, y_train_res, estimators)
        joblib.dump(models["voting"], voting_path)

    dnn_path = model_dir / "dnn.h5"
    if dnn_path.is_file():
        from tensorflow.keras.models import load_model
        models["dnn"] = load_model(dnn_path)
    else:
        _, dnn_model = deep_learning(X_train_res, y_train_res, X_test_scaled, y_test)
        dnn_model.save(dnn_path)
        models["dnn"] = dnn_model

    return models

def predict_with_ci(models: Dict[str, Any], X: pd.DataFrame, members: list = None, ci_z: float = 1.96):
    if members is None:
        members = ['xgb', 'rf', 'lr']

    probs = []
    for m in members:
        mdl = models[m]
        if hasattr(mdl, "predict_proba"):
            p = mdl.predict_proba(X)[:, 1]
        else:
            df = mdl.decision_function(X)
            p = 1/(1+np.exp(-df))
        probs.append(p)
    probs = np.vstack(probs)
    mean_prob = probs.mean(axis=0)
    std_prob = probs.std(axis=0, ddof=0)
    lower = (mean_prob - ci_z * std_prob).clip(0,1)
    upper = (mean_prob + ci_z * std_prob).clip(0,1)
    return mean_prob, lower, upper, probs

def make_shap_force_html(model, X_background: pd.DataFrame, X_instance: pd.DataFrame):
    bg = X_background.sample(n=min(50, len(X_background)), random_state=42)
    try:
        explainer = None
        shap_values = None
        expected_value = None

        if isinstance(model, (XGBClassifier, RandomForestClassifier, GradientBoostingClassifier)):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_instance)
            expected_value = explainer.expected_value
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
                expected_value = expected_value[1]

        elif isinstance(model, LogisticRegression):
            explainer = shap.LinearExplainer(model, bg, feature_dependence="independent")
            shap_values = explainer.shap_values(X_instance)
            expected_value = explainer.expected_value

        else:
            def predict_proba_fn(data):
                if hasattr(model, "predict_proba"):
                    return model.predict_proba(data)[:, 1]
                elif hasattr(model, "predict"):
                    preds = model.predict(data)
                    return preds.flatten()
                else:
                    return np.zeros(len(data))

            explainer = shap.KernelExplainer(predict_proba_fn, bg)
            shap_values = explainer.shap_values(X_instance, nsamples=100)
            expected_value = explainer.expected_value

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        fp = shap.force_plot(expected_value, shap_values[0,:], X_instance.iloc[0,:], matplotlib=False, show=False)
        html = f"<head>{shap.getjs()}</head><body>{fp.html()}</body>"
        return html
    except Exception as e:
        return f"<div>SHAP plotting failed: {e}</div>"

def dataframe_to_download_link(df: pd.DataFrame, filename: str = "predictions.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

st.title("Cardiovascular Risk — Clinical Insights")

with st.sidebar:
    st.header("Dataset & Model")
    data_path = st.text_input("Path to dataset CSV (semicolon-separated)", value="cardio_train.csv")
    if st.button("Load & Prepare Dataset"):
        st.rerun()
    st.markdown("**Note**: Change path to retrain.")

try:
    df_cleaned, X_train, y_train, X_test, y_test = load_and_prepare(data_path)
except Exception as e:
    st.error(f"Failed to load dataset from `{data_path}`: {e}")
    st.stop()

X_train_res, y_train_res, X_test_scaled, robust_scaler, standard_scaler = resample_and_scale(X_train, y_train, X_test)

numeric_robust = ["height", "weight", "ap_hi", "ap_lo"]
numeric_standard = ["age_years"]
for col in numeric_robust + numeric_standard:
    if col not in X_train_res.columns:
        st.error(f"Expected column `{col}` not found in training data. Check preprocess pipeline.")
        st.stop()

cluster_profile, outliers_count, outliers_ratio, kmeans_model = clustering_helper(X_train_res, y_train_res)
models = get_models_cached(X_train_res, y_train_res, X_test_scaled, y_test)

with st.expander("Model performance (holdout test set)"):
    X_test_copy = X_test_scaled

    st.write("Holdout test set size:", X_test_copy.shape[0])
    
    metrics_df = ensemble_evaluation(models, X_test_copy, y_test)
    st.dataframe(metrics_df.style.format({"Accuracy": "{:.4f}", "ROC AUC": "{:.4f}"}))

    st.write("Calibration curves (holdout):")
    curves = get_calibration_curves(models, X_test_copy, y_test)
    fig, ax = plt.subplots(figsize=(6,4))
    for name, (prob_true, prob_pred) in curves.items():
        if name == "dnn":
            probs = models[name].predict(X_test_copy).flatten()
        elif hasattr(models[name], "predict_proba"):
            probs = models[name].predict_proba(X_test_copy)[:, 1]
        else:
            probs = None
        
        auc_val = roc_auc_score(y_test, probs) if probs is not None else 0.5
        ax.plot(prob_pred, prob_true, marker='o', label=f"{name} (AUC={auc_val:.3f})")
        
    ax.plot([0,1], [0,1], linestyle='--', color='gray')
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("True Probability")
    ax.set_title("Calibration Curves (Holdout)")
    ax.legend()
    st.pyplot(fig)

st.header("Single patient assessment")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Enter clinical parameters")
    gender = st.selectbox("Gender (1 = female, 2 = male)", options=[1,2], index=0)
    height = st.number_input("Height (cm)", min_value=50, max_value=250, value=165)
    weight = st.number_input("Weight (kg)", min_value=20, max_value=300, value=70)
    ap_hi = st.number_input("Systolic BP (ap_hi)", min_value=50, max_value=300, value=120)
    ap_lo = st.number_input("Diastolic BP (ap_lo)", min_value=30, max_value=200, value=80)
    cholesterol = st.selectbox("Cholesterol (1: normal, 2: above normal, 3: well above)", [1,2,3], index=0)
    gluc = st.selectbox("Glucose (1: normal, 2: above, 3: well above)", [1,2,3], index=0)
    smoke = st.selectbox("Smoke (0/1)", [0,1], index=0)
    alco = st.selectbox("Alcohol intake (0/1)", [0,1], index=0)
    active = st.selectbox("Physical activity (0/1)", [0,1], index=1)
    age_years = st.number_input("Age (years)", min_value=1, max_value=120, value=45)

    predict_flag = False

    if st.button("Run prediction"):
        predict_flag = True

    if predict_flag:
        single = pd.DataFrame([{
            "gender": gender,
            "height": float(height),
            "weight": float(weight),
            "ap_hi": float(ap_hi),
            "ap_lo": float(ap_lo),
            "cholesterol": int(cholesterol),
            "gluc": int(gluc),
            "smoke": int(smoke),
            "alco": int(alco),
            "active": int(active),
            "age_years": int(age_years)
        }])
        single_scaled = single.copy()
        single_scaled[numeric_robust] = robust_scaler.transform(single_scaled[numeric_robust])
        single_scaled[numeric_standard] = standard_scaler.transform(single_scaled[numeric_standard])
        mean_prob, lower, upper, raw_probs = predict_with_ci(models, single_scaled, members=['xgb','rf','lr'], ci_z=1.96)
        prob = float(mean_prob[0])
        low = float(lower[0])
        high = float(upper[0])
        st.metric("Predicted cardiovascular risk (probability)", f"{prob:.3f}", delta=f"CI [{low:.3f}, {high:.3f}]")

        expl_model_choice = st.selectbox("Explain prediction with model", options=["xgb","rf","lr","voting","gb","rf_simple","dnn"], index=0)
        expl_model = models[expl_model_choice]

        with st.expander("SHAP explanation (force plot)"):
            try:
                html = make_shap_force_html(expl_model, X_train_res, single_scaled)
                st.components.v1.html(html, height=400, scrolling=True)
            except Exception as e:
                st.error(f"SHAP explanation failed: {e}")

        cluster_label = kmeans_model.predict(single_scaled)[0]
        st.write(f"Assigned cluster: **{int(cluster_label)}**")
        nn = NearestNeighbors(n_neighbors=6)
        nn.fit(X_train_res)
        distances, idxs = nn.kneighbors(single_scaled, n_neighbors=6)
        similar = X_train_res.iloc[idxs[0]].copy()
        nn_orig = NearestNeighbors(n_neighbors=6)
        nn_orig.fit(X_train)
        _, idxs_orig = nn_orig.kneighbors(single, n_neighbors=6)
        similar_orig = X_train.iloc[idxs_orig[0]].copy()
        st.write("Similar patients (from original training set):")
        st.dataframe(similar_orig.reset_index(drop=True))

st.header("Batch processing (CSV upload)")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    try:
        df_in = pd.read_csv(uploaded_file)
        required_cols = ["gender","age_years","height","weight","ap_hi","ap_lo","cholesterol","gluc","smoke","alco","active"]
        missing = [c for c in required_cols if c not in df_in.columns]
        if missing:
            st.error(f"Uploaded CSV is missing columns: {missing}")
        else:
            df_proc = df_in.copy()
            df_proc[numeric_robust] = robust_scaler.transform(df_proc[numeric_robust])
            df_proc[numeric_standard] = standard_scaler.transform(df_proc[numeric_standard])
            mean_prob, lower, upper, raw_probs = predict_with_ci(models, df_proc, members=['xgb','rf','lr'], ci_z=1.96)
            df_out = df_in.copy()
            df_out["pred_prob"] = mean_prob
            df_out["ci_lower"] = lower
            df_out["ci_upper"] = upper

            st.dataframe(df_out.head(50))
            st.markdown(dataframe_to_download_link(df_out, filename="batch_predictions.csv"), unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Failed to process uploaded CSV: {e}")

with st.expander("Advanced Diagnostics & Cluster Profile"):
    st.write("Cluster profile sample (from KMeans on resampled train):")
    st.dataframe(cluster_profile)
    st.write("You can retrain models by changing the dataset path.")
