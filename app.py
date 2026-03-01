import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─── Configuration de la page ───────────────────────────────────────────────
st.set_page_config(
    page_title="CreditIQ — Analyse de Crédit",
    page_icon="🏦",
    layout="wide"
)

# ─── CSS Personnalisé ────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: #F7F8FC;
        color: #1A1A2E;
    }

    .main-header {
        background: linear-gradient(135deg, #1A1A2E 0%, #16213E 60%, #0F3460 100%);
        padding: 2.5rem 3rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
    }

    .main-header h1 {
        font-family: 'Playfair Display', serif;
        font-size: 2.4rem;
        font-weight: 700;
        color: white;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .main-header p {
        color: #A8B2D8;
        font-size: 1rem;
        margin-top: 0.4rem;
        font-weight: 300;
    }

    .card {
        background: white;
        border-radius: 14px;
        padding: 1.8rem;
        box-shadow: 0 2px 16px rgba(0,0,0,0.06);
        border: 1px solid #E8ECF4;
        margin-bottom: 1.2rem;
    }

    .card h3 {
        font-family: 'Playfair Display', serif;
        font-size: 1.1rem;
        color: #1A1A2E;
        margin-bottom: 1rem;
        border-bottom: 2px solid #E8ECF4;
        padding-bottom: 0.6rem;
    }

    .result-approved {
        background: linear-gradient(135deg, #0F9B58, #00C896);
        border-radius: 14px;
        padding: 2rem;
        text-align: center;
        color: white;
    }

    .result-rejected {
        background: linear-gradient(135deg, #C0392B, #E74C3C);
        border-radius: 14px;
        padding: 2rem;
        text-align: center;
        color: white;
    }

    .result-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .result-proba {
        font-size: 3rem;
        font-weight: 700;
        letter-spacing: -1px;
    }

    .result-label {
        font-size: 0.9rem;
        opacity: 0.85;
        margin-top: 0.3rem;
    }

    .metric-box {
        background: #F0F4FF;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
        border-left: 4px solid #0F3460;
    }

    .metric-box .value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F3460;
    }

    .metric-box .label {
        font-size: 0.78rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .stSelectbox label, .stSlider label {
        font-weight: 500;
        color: #374151;
        font-size: 0.9rem;
    }

    div[data-testid="stButton"] button {
        background: linear-gradient(135deg, #1A1A2E, #0F3460);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 2rem;
        font-size: 1rem;
        font-weight: 500;
        width: 100%;
        cursor: pointer;
        transition: opacity 0.2s;
        font-family: 'DM Sans', sans-serif;
    }

    div[data-testid="stButton"] button:hover {
        opacity: 0.88;
    }

    .footer {
        text-align: center;
        color: #9CA3AF;
        font-size: 0.8rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #E8ECF4;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏦 CreditIQ</h1>
    <p>Système d'analyse prédictive de remboursement de crédit</p>
</div>
""", unsafe_allow_html=True)

# ─── Chargement du modèle ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("xgb_credit_model.pkl")

model = load_model()

# ─── Layout principal ────────────────────────────────────────────────────────
col_form, col_result = st.columns([1, 1], gap="large")

with col_form:
    st.markdown('<div class="card"><h3>📋 Informations du Demandeur</h3>', unsafe_allow_html=True)

    employment_status = st.selectbox(
        "Statut d'emploi",
        options=["Employed", "Self-Employed", "Unemployed", "Retired"],
        format_func=lambda x: {
            "Employed": "👔 Salarié",
            "Self-Employed": "💼 Indépendant",
            "Unemployed": "🔍 Sans emploi",
            "Retired": "🧓 Retraité"
        }[x]
    )

    education_level = st.selectbox(
        "Niveau d'éducation",
        options=["High School", "Bachelor's", "Master's", "PhD"],
        format_func=lambda x: {
            "High School": "🎓 Lycée",
            "Bachelor's": "🎓 Licence",
            "Master's": "🎓 Master",
            "PhD": "🎓 Doctorat"
        }[x]
    )

    interest_rate = st.slider(
        "Taux d'intérêt (%)",
        min_value=1.0,
        max_value=30.0,
        value=10.0,
        step=0.5,
        format="%.1f%%"
    )

    st.markdown('</div>', unsafe_allow_html=True)

    predict_btn = st.button("🔍 Analyser le dossier")

with col_result:

    if predict_btn:
        # Préparer les données
        input_data = pd.DataFrame([{
            "employment_status": employment_status,
            "interest_rate": interest_rate,
            "education_level": education_level
        }])

        # Prédiction
        proba = model.predict_proba(input_data)[0][1]
        decision = proba >= 0.5

        # Affichage du résultat
        if decision:
            st.markdown(f"""
            <div class="result-approved">
                <div class="result-title">✅ Crédit Approuvé</div>
                <div class="result-proba">{proba*100:.1f}%</div>
                <div class="result-label">Probabilité de remboursement</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-rejected">
                <div class="result-title">❌ Crédit Refusé</div>
                <div class="result-proba">{proba*100:.1f}%</div>
                <div class="result-label">Probabilité de remboursement</div>
            </div>
            """, unsafe_allow_html=True)

        # Métriques
        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="value">{proba*100:.1f}%</div>
                <div class="label">Remboursement</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="value">{(1-proba)*100:.1f}%</div>
                <div class="label">Risque défaut</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            risque = "Faible 🟢" if proba > 0.7 else "Moyen 🟡" if proba > 0.5 else "Élevé 🔴"
            st.markdown(f"""
            <div class="metric-box">
                <div class="value" style="font-size:1.1rem">{risque}</div>
                <div class="label">Niveau de risque</div>
            </div>""", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="card" style="text-align:center; padding: 3rem;">
            <div style="font-size: 3rem;">📂</div>
            <div style="font-family: 'Playfair Display', serif; font-size: 1.2rem; color: #1A1A2E; margin-top: 1rem;">
                En attente d'analyse
            </div>
            <div style="color: #9CA3AF; font-size: 0.9rem; margin-top: 0.5rem;">
                Remplissez le formulaire et cliquez sur "Analyser le dossier"
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─── Features Importances ────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="card"><h3>📊 Importance des Variables</h3>', unsafe_allow_html=True)

try:
    feature_names = model[:-1].get_feature_names_out()
    importances = model[-1].feature_importances_

    feat_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=True).tail(10)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    colors = ["#0F3460" if i == len(feat_df)-1 else "#A8B2D8" for i in range(len(feat_df))]
    bars = ax.barh(feat_df["feature"], feat_df["importance"], color=colors, height=0.6)
    ax.set_xlabel("Importance", fontsize=9, color="#6B7280")
    ax.tick_params(axis='both', labelsize=8, colors="#374151")
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color('#E8ECF4')
    ax.set_facecolor('#FFFFFF')
    fig.patch.set_facecolor('#FFFFFF')
    plt.tight_layout()
    st.pyplot(fig)

except Exception as e:
    st.info("ℹ️ Chargez un modèle valide pour afficher les importances.")

st.markdown('</div>', unsafe_allow_html=True)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    CreditIQ • Modèle XGBoost • Analyse prédictive de crédit
</div>
""", unsafe_allow_html=True)