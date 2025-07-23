import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# Load model and preprocessors
model = joblib.load("ovarian_cancer_ensemble_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.set_page_config(page_title="Ovarian Cancer Predictor", layout="wide")

# ---- Sidebar ----
st.sidebar.title("🧬 Risk Factor Inputs")
st.sidebar.markdown("Enter the patient's clinical and genetic risk factors.")

# Input form
age = st.sidebar.slider("Age", 20, 90, 45)
family_history = st.sidebar.selectbox("Family History of Ovarian Cancer", ["Yes", "No"])
brca_mutation = st.sidebar.selectbox("BRCA Mutation", ["Positive", "Negative", "Unknown"])
hormone_therapy = st.sidebar.selectbox("Hormone Therapy", ["Yes", "No"])
endometriosis = st.sidebar.selectbox("Endometriosis", ["Yes", "No"])
infertility = st.sidebar.selectbox("Infertility", ["Yes", "No"])
obesity = st.sidebar.selectbox("Obesity", ["Yes", "No"])
smoking = st.sidebar.selectbox("Smoking", ["Yes", "No"])
menopause_age = st.sidebar.slider("Age at Menopause", 40, 60, 50)
first_pregnancy_age = st.sidebar.slider("Age at First Pregnancy", 15, 45, 25)
number_of_pregnancies = st.sidebar.slider("Number of Pregnancies", 0, 6, 2)

# ---- Main UI ----
st.title("🔍 Ovarian Cancer Prediction System")
st.markdown("This system predicts the likelihood of ovarian cancer using a trained ensemble model based on clinical and lifestyle risk factors.")

if st.button("Predict Risk"):
    # Collect inputs into a DataFrame
    input_data = pd.DataFrame([[
        age,
        family_history,
        brca_mutation,
        hormone_therapy,
        endometriosis,
        infertility,
        obesity,
        smoking,
        menopause_age,
        first_pregnancy_age,
        number_of_pregnancies
    ]], columns=[
        "Age", "Family_History", "BRCA_Mutation", "Hormone_Therapy", "Endometriosis",
        "Infertility", "Obesity", "Smoking", "Menopause_Age", "First_Pregnancy_Age",
        "Number_of_Pregnancies"
    ])

    # Encode categorical features
    for col in ["Family_History", "BRCA_Mutation", "Hormone_Therapy",
                "Endometriosis", "Infertility", "Obesity", "Smoking"]:
        le = label_encoders[col]
        input_data[col] = le.transform(input_data[col])

    # Scale numerical features
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # Display result
    if prediction == 1:
        st.error(f"⚠️ High Risk Detected — Probability: {probability:.2%}")
    else:
        st.success(f"✅ Low Risk Detected — Probability: {probability:.2%}")

    st.markdown("---")
    st.subheader("📊 Risk Factor Profile (Radar Chart)")

    # Prepare data for radar chart
    raw_input = input_data.iloc[0].copy()
    raw_input_values = []

    display_labels = {
        "Age": age,
        "Family_History": family_history,
        "BRCA_Mutation": brca_mutation,
        "Hormone_Therapy": hormone_therapy,
        "Endometriosis": endometriosis,
        "Infertility": infertility,
        "Obesity": obesity,
        "Smoking": smoking,
        "Menopause_Age": menopause_age,
        "First_Pregnancy_Age": first_pregnancy_age,
        "Number_of_Pregnancies": number_of_pregnancies
    }

    for key, val in display_labels.items():
        if isinstance(val, str):
            raw_input_values.append(str(val))
        else:
            raw_input_values.append(val)

    categories = list(display_labels.keys())
    values = [raw_input[key] if isinstance(raw_input[key], (int, float)) else 1 for key in categories]

    # Normalize for radar chart scale
    max_values = [90, 1, 2, 1, 1, 1, 1, 1, 60, 45, 6]
    radar_values = [v / m if m else 0 for v, m in zip(values, max_values)]

    # Radar chart
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=radar_values,
        theta=categories,
        fill='toself',
        name='Risk Profile',
        line_color='crimson'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        template="plotly_dark" if prediction == 1 else "plotly_white",
        title="Risk Factor Radar Chart"
    )

    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown(
    """
    <hr>
    <center>
    <small>Developed for educational and predictive modeling purposes. Not a substitute for clinical diagnosis.</small>
    </center>
    """,
    unsafe_allow_html=True
)
