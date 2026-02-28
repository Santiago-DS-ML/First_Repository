import streamlit as st
import joblib
import numpy as np
import pandas as pd

# -------------------------
# Configuration page
# -------------------------
st.set_page_config(page_title="Credit Risk Scoring", layout="centered")

st.title("Système de Scoring Crédit")
st.markdown("Modèle basé sur XGBoost")

# -------------------------
# Charger modèle
# -------------------------
model = joblib.load("xgb_credit_model.pkl")

# Si scaler utilisé :
# scaler = joblib.load("scaler.pkl")

# -------------------------
# Sidebar - Inputs
# -------------------------
st.sidebar.header("Informations Client")

employment_status = st.sidebar.selectbox(
    "Statut professionnel",
    ['Self-employed', 'Employed', 'Unemployed', 'Retired', 'Student']
)

interest_rate = st.sidebar.slider(
    "Taux d'intérêt (%)",
    0.0, 30.0, 10.0
)

education_level = st.sidebar.selectbox(
    "Niveau d'éducation",
    ['High School', "Master's", "Bachelor's", 'PhD', 'Other']
)

# -------------------------
# Prédiction
# -------------------------
if st.button("Analyser le risque"):

    data = pd.DataFrame({
        "employment_status": [employment_status],
        "interest_rate": [interest_rate],
        "education_level": [education_level]
    })

    # Si scaler :
    # data = scaler.transform(data)

    probability = model.predict_proba(data)[0][1]

    st.subheader("Résultat du scoring")

    st.metric("Probabilité de remboursement", f"{round(probability*100,2)} %")

    if probability <= 0.5:
        st.error("Probabilité de remboursement faible - Refus recommandé")
    elif probability > 0.6:
         st.success("Probabilité de remboursement élevée - Bon crédit")
    else:
        st.error("Probabilité de remboursement faible - Refus recommandé")
