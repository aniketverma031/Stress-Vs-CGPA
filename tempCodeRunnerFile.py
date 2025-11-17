# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import base64

st.set_page_config(page_title="Stress vs CGPA Predictor", layout="centered")

# ---------------------------------------------------
# Convert Local PNG/JPG to Base64 for Background
# ---------------------------------------------------
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Load your local background image
base64_bg = get_base64_image("stress.png")   # ⭐ Make sure stress.png is in the same folder

# ---------------------------------------------------
# DYNAMIC MOVING BACKGROUND CSS
# ---------------------------------------------------
st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{base64_bg}");
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
        background-position: center;
        animation: moveBg 40s linear infinite;
    }}

    @keyframes moveBg {{
        0% {{ background-position: 0% 0%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 0%; }}
    }}

    .block-container {{
        background: rgba(0,0,0,0.55);
        padding: 2rem;
        border-radius: 12px;
    }}

    h1, h2, h3, label, div, p {{
        color: white !important;
    }}
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Helper: load artifact
# ---------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifact(path="StressVsCGPA_FinalModel.pkl"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model artifact not found at {path}. Place the .pkl in the app folder.")
    artifact = joblib.load(path)
    model = artifact.get("model")
    scaler = artifact.get("scaler")
    columns = list(artifact.get("columns"))
    return model, scaler, columns

# ---------------------------------------------------
# Prediction utilities
# ---------------------------------------------------
def prepare_input(df_input, feature_columns, scaler):
    df_enc = pd.get_dummies(df_input, drop_first=True)
    for c in feature_columns:
        if c not in df_enc.columns:
            df_enc[c] = 0
    df_enc = df_enc[feature_columns]

    if scaler is not None:
        num_cols = [c for c in feature_columns if c in ("AGE", "Stress level")]
        if len(num_cols) > 0:
            df_enc[num_cols] = scaler.transform(df_enc[num_cols])
    return df_enc

def predict(model, X):
    pred = model.predict(X)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        classes = model.classes_
    else:
        classes = np.array(["Low", "Medium", "High"])
    return pred, proba, classes

# ---------------------------------------------------
# UI
# ---------------------------------------------------
st.title("Stress vs CGPA — Predictor")
st.markdown(
    "Enter the student details below. This app uses a trained ML model to predict the CGPA category "
    "(Low / Medium / High)."
)

try:
    model, scaler, feature_columns = load_artifact()
except Exception as e:
    st.error(str(e))
    st.stop()

with st.form("input_form"):
    st.subheader("Student Inputs")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=15, max_value=40, value=20, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        year = st.selectbox("Year of Study", ["1st", "2nd", "3rd", "4th"])
    with col2:
        stress = st.slider("Stress Level (1 = low, 10 = high)", min_value=1, max_value=10, value=5)
        social = st.selectbox("Social Media Impact on Academics", ["Low", "Medium", "High"])

    submit = st.form_submit_button("Predict")

if submit:
    row = {
        "AGE": age,
        "Stress level": stress,
        "GENDER": gender,
        "Year of Study": year,
        "Social Media Impact on Academics": social
    }

    input_df = pd.DataFrame([row])
    X_input = prepare_input(input_df, feature_columns, scaler)

    pred_label, proba, classes = predict(model, X_input)

    st.markdown("### Prediction")
    st.success(f"Predicted CGPA Category: **{pred_label}**")

    if proba is not None:
        proba_df = pd.DataFrame({
            "Category": classes,
            "Probability": [float(x) for x in proba]
        }).sort_values("Probability", ascending=False)
        st.table(proba_df.style.format({"Probability": "{:.3f}"}))

st.markdown("---")
