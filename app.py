# app.py
"""
Stress vs CGPA — Predictor & Suggestions
Requirements: streamlit, pandas, numpy, joblib
Place `stress.png` and `StressVsCGPA_FinalModel.pkl` in the same folder (or /mnt/data/).
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import base64
from typing import List, Tuple

st.set_page_config(page_title="Stress vs CGPA Predictor", layout="centered", initial_sidebar_state="auto")

# -------------------------
# Helper: base64 background
# -------------------------
def get_base64_image(image_path: str) -> str:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Background image not found at {image_path}")
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Attempt local file paths (current folder or /mnt/data)
BG_CANDIDATES = ["stress.png", "/mnt/data/stress.png"]
bg_path = next((p for p in BG_CANDIDATES if os.path.exists(p)), None)
if bg_path is None:
    st.error("Background image `stress.png` not found. Place it in the app folder or /mnt/data/")
    st.stop()

base64_bg = get_base64_image(bg_path)

# -------------------------
# Style & animated background
# -------------------------
st.markdown(
    f"""
    <style>
    /* ensure top content not cut off and background visible */
    .stApp {{
        background-image: url("data:image/png;base64,{base64_bg}");
        background-size: cover;                 /* fill the page */
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
        animation: moveBg 45s linear infinite;
    }}

    /* subtle movement */
    @keyframes moveBg {{
        0% {{ background-position: 10% 10%; transform: scale(1); }}
        50% {{ background-position: 90% 40%; transform: scale(1.02); }}
        100% {{ background-position: 10% 10%; transform: scale(1); }}
    }}

    /* add top padding so heading isn't cut by the browser toolbar */
    .stApp > header ~ div {{
        padding-top: 32px !important;
    }}

    /* translucent card for content so background is visible */
    .block-container {{
        background: rgba(6, 8, 10, 0.55) !important;
        border-radius: 12px;
        padding: 28px 32px !important;
        box-shadow: 0 8px 30px rgba(0,0,0,0.6);
    }}

    /* form fields look */
    .stButton>button {{
        border-radius: 10px;
        padding: .6rem 1rem;
    }}

    /* floating CGPA card on the right */
    .floating-cgpa {{
        position: fixed;
        right: 36px;
        top: 120px;
        background: rgba(10,10,12,0.6);
        color: #e6f7ff;
        border-radius: 12px;
        padding: 18px 22px;
        min-width: 180px;
        text-align: center;
        backdrop-filter: blur(6px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.6);
        z-index: 9999;
        border: 1px solid rgba(255,255,255,0.04);
    }}

    .floating-cgpa .big {{
        font-size: 28px;
        font-weight: 700;
    }}

    .floating-cgpa .small {{
        font-size: 12px;
        opacity: 0.85;
    }}

    /* ensure text is readable */
    h1, h2, h3, label, p, .stMarkdown {{
        color: #fff !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------
# Load model artifact
# -------------------------
@st.cache_resource(show_spinner=False)
def load_artifact(path_candidates: List[str] = None):
    if path_candidates is None:
        path_candidates = ["StressVsCGPA_FinalModel.pkl", "/mnt/data/StressVsCGPA_FinalModel.pkl"]
    found = next((p for p in path_candidates if os.path.exists(p)), None)
    if found is None:
        raise FileNotFoundError("Model artifact not found. Place StressVsCGPA_FinalModel.pkl in the app folder or /mnt/data/")
    artifact = joblib.load(found)
    model = artifact.get("model", None)
    scaler = artifact.get("scaler", None)
    columns = list(artifact.get("columns", []))
    return model, scaler, columns

try:
    model, scaler, feature_columns = load_artifact()
except Exception as e:
    st.error(str(e))
    st.stop()

# -------------------------
# Prediction utilities
# -------------------------
def prepare_input(df_input: pd.DataFrame, feature_columns: List[str], scaler):
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

def predict(model, X: pd.DataFrame) -> Tuple[str, np.ndarray, np.ndarray]:
    pred = model.predict(X)[0]
    proba = None
    classes = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        classes = model.classes_
    else:
        classes = np.array(["Low", "Medium", "High"])
    return pred, proba, classes

# -------------------------
# Suggestions engine
# -------------------------
def generate_suggestions(row: dict, pred_label: str) -> List[str]:
    suggestions = []
    stress = int(row.get("Stress level", 5))
    social = row.get("Social Media Impact on Academics", "Low")
    age = row.get("AGE", None)
    year = row.get("Year of Study", "")

    # Base suggestions by stress
    if stress >= 8:
        suggestions += [
            "🧠 High stress detected — take short breaks every 45–60 minutes (5–10 min).",
            "😴 Prioritise sleep: aim for 7–8 hours nightly; poor sleep worsens performance.",
            "📞 Consider reaching out to a counselor or speak with a trusted mentor.",
            "📵 Reduce late-night screen time; use focus apps or airplane mode during study blocks."
        ]
    elif 5 <= stress <= 7:
        suggestions += [
            "🗂️ Moderate stress — make a weekly study plan with short goals and clear priorities.",
            "🏃 Add short physical activity (15–25 min walk/exercise) to reduce tension.",
            "⏱ Use Pomodoro (25/5) to keep focus and avoid burnout."
        ]
    else:
        suggestions += [
            "✅ Low stress — keep the healthy routines: consistent sleep, breaks, and practice.",
            "📚 Keep using active recall and spaced repetition to maintain performance."
        ]

    # Social media impact
    if social == "High":
        suggestions.append("🔕 Social apps are impacting academics — schedule fixed social time; enable app timers.")
    elif social == "Medium":
        suggestions.append("🕒 Moderate social use — try blocking distracting apps while studying.")

    # Year-based tips
    if "1st" in year:
        suggestions.append("🤝 First year tip: join a study group or seek peer mentoring to adapt faster.")
    elif "4th" in year:
        suggestions.append("🎯 Final year tip: focus on capstone/project milestones and placement prep.")

    # Predicted label based tips
    if pred_label == "Low":
        suggestions += [
            "📈 Focus on fundamentals: identify weak subjects and practice solved problems.",
            "👩‍🏫 Consider tutoring or small group sessions for difficult topics."
        ]
    elif pred_label == "High":
        suggestions += [
            "🏆 Great — maintain consistency and consider helping peers (teaching reinforces learning).",
            "📖 Try advanced/competitive problems or research topics to grow further."
        ]
    else:
        suggestions += ["⚖️ Seek balance between practice and rest to move towards a higher CGPA bracket."]

    # De-duplicate and return
    seen = set()
    final = []
    for s in suggestions:
        if s not in seen:
            final.append(s)
            seen.add(s)
    return final

# -------------------------
# App UI
# -------------------------
st.markdown("<h1 style='margin-bottom:0.1rem'>🧠 Stress vs CGPA — Predictor & Suggestions</h1>", unsafe_allow_html=True)
st.markdown("<p style='margin-top:0.1rem; color: #d6d6d6;'>Enter student details to get a predicted CGPA category and personalised suggestions.</p>", unsafe_allow_html=True)
st.markdown("---")

with st.form("input_form"):
    st.subheader("Student Inputs")

    col1, col2 = st.columns([1, 1])
    with col1:
        age = st.number_input("Age", min_value=15, max_value=40, value=20, step=1)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        year = st.selectbox("Year of Study", ["1st", "2nd", "3rd", "4th"])
    with col2:
        stress = st.slider("Stress Level (1 = low, 10 = high)", min_value=1, max_value=10, value=5)
        social = st.selectbox("Social Media Impact on Academics", ["Low", "Medium", "High"])
        # small helper text
        st.caption("Be honest — suggestions will be more useful when inputs reflect reality.")

    submit = st.form_submit_button("Predict & Suggest")

# Show quick indicator even before submit
st.markdown("### Quick indicators")
st.write("Stress progress (visual)")
st.progress(int((stress / 10) * 100))

# Process on submit
if submit:
    # Prepare input row and dataframe
    row = {
        "AGE": age,
        "Stress level": stress,
        "GENDER": gender,
        "Year of Study": year,
        "Social Media Impact on Academics": social
    }
    input_df = pd.DataFrame([row])
    try:
        X_input = prepare_input(input_df, feature_columns, scaler)
    except Exception as e:
        st.error(f"Error preparing input: {e}")
        st.stop()

    try:
        pred_label, proba, classes = predict(model, X_input)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    # Show prediction and probabilities
    st.markdown("### Prediction")
    st.success(f"Predicted CGPA Category: **{pred_label}**")

    if proba is not None:
        proba_df = pd.DataFrame({
            "Category": classes,
            "Probability": [float(x) for x in proba]
        }).sort_values("Probability", ascending=False)
        st.table(proba_df.style.format({"Probability": "{:.3f}"}))

    # Generate suggestions
    suggestions = generate_suggestions(row, pred_label)
    st.markdown("### AI-Generated Personalized Suggestions")
    for s in suggestions:
        st.markdown(f"- {s}")

    # Show floating CGPA-like indicator (mock numeric based on class)
    # If model provides numeric cgpa prediction instead of label, you can replace this block.
    cgpa_estimate_map = {"Low": 5.5, "Medium": 6.8, "High": 8.5}
    cgpa_value = cgpa_estimate_map.get(pred_label, 6.5)

    st.markdown(
        f"""
        <div class="floating-cgpa">
            <div class="small">Estimated CGPA</div>
            <div class="big">{cgpa_value:.2f}</div>
            <div class="small">Confidence: {(max(proba) if proba is not None else 0.0):.2%}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

else:
    st.info("Fill the form and click **Predict & Suggest** to get a prediction and personalized tips.")

st.markdown("---")
st.markdown("<div style='font-size:12px;color:#cfcfcf'>Tip: Adjust the stress slider and social media option to see different suggestions.</div>", unsafe_allow_html=True)
