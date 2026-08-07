import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# ---------- Page Config ----------
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered"
)

# ---------- Load Model ----------
@st.cache_resource
def load_model():
    return joblib.load("heart_disease_pipeline.pkl")

model = load_model()

# ---------- Header ----------
st.title("❤️ Heart Disease Risk Predictor")
st.markdown(
    "This app uses a **Logistic Regression** model trained on the "
    "UCI Heart Disease dataset (88% test accuracy) to estimate the "
    "likelihood of heart disease based on patient health metrics."
)
st.divider()

# ---------- Input Form ----------
st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 20, 100, 50)
    sex = st.radio("Sex", ["Male", "Female"], horizontal=True)
    cp = st.selectbox(
        "Chest Pain Type",
        [0, 1, 2, 3],
        format_func=lambda x: {
            0: "Typical Angina", 1: "Atypical Angina",
            2: "Non-anginal Pain", 3: "Asymptomatic"
        }[x]
    )
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 220, 120)
    chol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl?", ["No", "Yes"], horizontal=True)
    restecg = st.selectbox(
        "Resting ECG Result",
        [0, 1, 2],
        format_func=lambda x: {0: "Normal", 1: "ST-T Abnormality", 2: "LV Hypertrophy"}[x]
    )

with col2:
    thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.radio("Exercise Induced Angina?", ["No", "Yes"], horizontal=True)
    oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 10.0, 1.0, step=0.1)
    slope = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        [0, 1, 2],
        format_func=lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x]
    )
    ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
    thal = st.selectbox(
        "Thalassemia",
        [1, 2, 3],
        format_func=lambda x: {1: "Fixed Defect", 2: "Normal", 3: "Reversible Defect"}[x]
    )

st.divider()

# ---------- Predict ----------
if st.button("🔍 Predict Risk", use_container_width=True, type="primary"):
    input_df = pd.DataFrame([{
        "age": age,
        "sex": 1 if sex == "Male" else 0,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": 1 if fbs == "Yes" else 0,
        "restecg": restecg,
        "thalach": thalach,
        "exang": 1 if exang == "Yes" else 0,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    st.subheader("Result")

    if prediction == 1:
        st.error(f"⚠️ **High risk of heart disease**")
    else:
        st.success(f"✅ **Low risk of heart disease**")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba * 100,
        number={'suffix': "%"},
        title={'text': "Heart Disease Risk"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#d62728" if proba > 0.5 else "#2ca02c"},
            'steps': [
                {'range': [0, 40], 'color': "#e6f4ea"},
                {'range': [40, 70], 'color': "#fff3cd"},
                {'range': [70, 100], 'color': "#f8d7da"},
            ],
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("See input summary"):
        st.dataframe(input_df.T.rename(columns={0: "Value"}))

st.divider()
st.caption(
    "⚠️ This tool is for educational/demo purposes only and is **not** a substitute "
    "for professional medical advice, diagnosis, or treatment."
)
