"""
StormCast: AI-Powered Storm Risk Prediction Application
Streamlit Application for Live Meteorological Inference.
"""

import os
import sys
import joblib
import pandas as pd
import streamlit as st

# Ensure project root is on sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import get_season
from src.weather_service import (
    fetch_live_weather, build_prediction_features,
    US_STATE_MAP, TIMEZONE_MAP
)

# Page configuration
st.set_page_config(
    page_title="StormCast | Storm Risk Prediction",
    page_icon="⛈️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom Styling for clean, modern presentation
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #1e293b;
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 4px;
    }
    .sub-title {
        text-align: center;
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 24px;
    }
    .disclaimer-box {
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 10px 16px;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #475569;
        margin-bottom: 24px;
    }
    .weather-card {
        background: linear-gradient(135deg, #f0fdf4 0%, #e0f2fe 100%);
        border: 1px solid #bae6fd;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    }
    .weather-val {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0369a1;
    }
    .weather-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .storm-banner-danger {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-top: 15px;
        margin-bottom: 20px;
    }
    .storm-banner-safe {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 2px solid #22c55e;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-top: 15px;
        margin-bottom: 20px;
    }
    .storm-result-title {
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
    }
    .prob-number {
        font-size: 3rem;
        font-weight: 900;
        margin: 8px 0;
    }
    .risk-badge {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load trained Scikit-Learn pipelines from disk."""
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    occ_path = os.path.join(models_dir, "storm_occurrence_model.joblib")
    sev_path = os.path.join(models_dir, "storm_severity_model.joblib")

    models = {}
    if os.path.exists(occ_path):
        models['occurrence'] = joblib.load(occ_path)
    if os.path.exists(sev_path):
        models['severity'] = joblib.load(sev_path)
    return models


def determine_risk_tier(prob: float) -> str:
    """Categorize probability into standard risk level."""
    if prob >= 0.65:
        return "HIGH"
    elif prob >= 0.35:
        return "MODERATE"
    else:
        return "LOW"


def main():
    # Application Title
    st.markdown('<div class="main-title">⛈️ StormCast</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">AI-Powered Storm Risk Prediction</div>', unsafe_allow_html=True)

    # Scientific Scope Note
    st.markdown(
        '<div class="disclaimer-box">'
        '<strong>Scientific Scope:</strong> StormCast uses live weather observations and a trained machine-learning '
        'model to estimate storm occurrence risk for the selected location.'
        '</div>',
        unsafe_allow_html=True
    )

    # Load machine learning models
    models = load_models()
    occ_model = models.get('occurrence')
    sev_model = models.get('severity')

    if occ_model is None:
        st.error("⚠️ Trained occurrence model not found in `models/`. Please verify the model file exists.")
        return

    # Initialize Session State
    if 'weather_data' not in st.session_state:
        st.session_state.weather_data = None
    if 'prediction_done' not in st.session_state:
        st.session_state.prediction_done = False
    if 'last_location' not in st.session_state:
        st.session_state.last_location = ""
    if 'pred_results' not in st.session_state:
        st.session_state.pred_results = None

    # Location Input Section
    st.markdown("### 📍 Location")

    loc_col1, loc_col2 = st.columns([3, 1])
    with loc_col1:
        city_input = st.text_input(
            "Enter City / Location",
            value=st.session_state.last_location if st.session_state.last_location else "Austin, Texas",
            placeholder="e.g. Austin, Dallas, Denver, Miami, Seattle, Delhi",
            label_visibility="collapsed"
        )
    with loc_col2:
        fetch_clicked = st.button("📍 Get Weather", use_container_width=True)

    # Location suggestions for quick demonstration
    st.caption("Quick test locations: "
               "[**Dallas**](javascript:void(0)) | "
               "[**Denver**](javascript:void(0)) | "
               "[**Miami**](javascript:void(0)) | "
               "[**Seattle**](javascript:void(0))")

    # API Fetch Action
    if fetch_clicked:
        if not city_input.strip():
            st.warning("Please enter a location name before fetching weather.")
        else:
            with st.spinner(f"Retrieving current weather for '{city_input}'..."):
                res = fetch_live_weather(city_input.strip())
                if res.get('success'):
                    st.session_state.weather_data = res
                    st.session_state.last_location = city_input.strip()
                    st.session_state.prediction_done = False
                    st.session_state.pred_results = None
                else:
                    st.session_state.weather_data = None
                    st.error(f"❌ {res.get('error', 'Unable to retrieve live weather data.')}")

    # Display Weather Summary Card if available
    w = st.session_state.weather_data
    if w and w.get('success'):
        st.markdown("---")
        st.markdown("### 🌦️ Current Weather")
        
        # Location header line
        loc_display = f"{w['city']}"
        if w.get('region') and w['region'] != w['city']:
            loc_display += f", {w['region']}"
        if w.get('country'):
            loc_display += f", {w['country']}"

        st.markdown(f"**📍 Location:** {loc_display} &nbsp;&nbsp;|&nbsp;&nbsp; "
                    f"**Condition:** {w.get('condition', 'N/A')} &nbsp;&nbsp;|&nbsp;&nbsp; "
                    f"**Time:** {w.get('observation_time', 'N/A')}")

        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Temperature", f"{w['temperature_c']} °C")
        with m2:
            st.metric("Precipitation", f"{w['precipitation_mm']} mm")
        with m3:
            st.metric("Humidity", f"{w['humidity']} %")
        with m4:
            st.metric("Wind Speed", f"{w['wind_kmh']} km/h")

        st.caption(f"Data source: {w.get('provider', 'Live Meteorological API')} "
                   f"(Coordinates: {w['latitude']:.2f}°N, {w['longitude']:.2f}°W)")

        # Prediction Trigger Button
        st.markdown("### 🤖 Storm Prediction")
        predict_clicked = st.button("⛈️ Predict Storm Risk", type="primary", use_container_width=True)

        if predict_clicked:
            with st.spinner("Transforming live data and executing Scikit-Learn model pipeline..."):
                # Feature Mapping
                features_df = build_prediction_features(w)

                # ML Occurrence Model Inference
                storm_pred = int(occ_model.predict(features_df)[0])
                storm_prob = float(occ_model.predict_proba(features_df)[0][1])
                risk_tier = determine_risk_tier(storm_prob)

                # ML Severity Model Inference
                sev_pred = None
                sev_probs = None
                if sev_model is not None:
                    try:
                        sev_pred = str(sev_model.predict(features_df)[0])
                        classes = sev_model.classes_
                        probs = sev_model.predict_proba(features_df)[0]
                        sev_probs = {c: round(float(p) * 100, 1) for c, p in zip(classes, probs)}
                    except Exception:
                        sev_pred = None

                st.session_state.prediction_done = True
                st.session_state.pred_results = {
                    'storm_pred': storm_pred,
                    'storm_prob': storm_prob,
                    'risk_tier': risk_tier,
                    'sev_pred': sev_pred,
                    'sev_probs': sev_probs,
                    'features': features_df.to_dict(orient='records')[0]
                }

    # Display Prediction Results
    if st.session_state.prediction_done and st.session_state.pred_results:
        res = st.session_state.pred_results
        storm_pred = res['storm_pred']
        storm_prob = res['storm_prob']
        risk_tier = res['risk_tier']
        sev_pred = res['sev_pred']
        sev_probs = res['sev_probs']

        st.markdown("---")

        if storm_pred == 1:
            # Storm Detected Card
            st.markdown(f"""
            <div class="storm-banner-danger">
                <div class="storm-result-title" style="color: #dc2626;">⛈️ STORM DETECTED</div>
                <div class="weather-label" style="margin-top: 10px;">Estimated Storm Probability</div>
                <div class="prob-number" style="color: #b91c1c;">{storm_prob*100:.1f}%</div>
                <div class="risk-badge" style="background-color: #ef4444; color: white;">RISK LEVEL: {risk_tier}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # No Storm Detected Card
            st.markdown(f"""
            <div class="storm-banner-safe">
                <div class="storm-result-title" style="color: #16a34a;">☀️ NO STORM DETECTED</div>
                <div class="weather-label" style="margin-top: 10px;">Estimated Storm Probability</div>
                <div class="prob-number" style="color: #15803d;">{storm_prob*100:.1f}%</div>
                <div class="risk-badge" style="background-color: #22c55e; color: white;">RISK LEVEL: {risk_tier}</div>
            </div>
            """, unsafe_allow_html=True)

        # Weather Severity Section
        if sev_pred:
            st.markdown("#### Predicted Weather Severity")
            sev_badge_color = {
                'Light': '#22c55e',
                'Moderate': '#f59e0b',
                'Heavy': '#ea580c',
                'Severe': '#dc2626'
            }.get(sev_pred, '#0284c7')

            st.markdown(
                f"<div style='font-size: 1.5rem; font-weight: 800; color: {sev_badge_color}; margin-bottom: 8px;'>"
                f"{sev_pred.upper()}</div>",
                unsafe_allow_html=True
            )

            if sev_probs:
                p_cols = st.columns(4)
                classes_order = ['Light', 'Moderate', 'Heavy', 'Severe']
                for idx, c in enumerate(classes_order):
                    if c in sev_probs:
                        with p_cols[idx]:
                            st.metric(label=c, value=f"{sev_probs[c]}%")

    # Manual Fallback Mode (Section 12 of Requirements)
    st.markdown("---")
    with st.expander("⚙️ Manual Prediction Mode (Fallback / Offline)"):
        st.markdown(
            "Use manual prediction mode if you are offline, testing a hypothetical scenario, "
            "or if the weather API cannot be reached."
        )

        with st.form("manual_prediction_form"):
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                man_lat = st.number_input("Latitude (°N)", min_value=24.0, max_value=50.0, value=32.77, step=0.01)
                man_lon = st.number_input("Longitude (°W)", min_value=-125.0, max_value=-66.0, value=-96.79, step=0.01)
                man_precip_in = st.number_input("Precipitation (inches)", min_value=0.0, max_value=15.0, value=0.0, step=0.05)
            with fc2:
                man_month = st.slider("Month of Year", min_value=1, max_value=12, value=7)
                man_day = st.slider("Day of Month", min_value=1, max_value=31, value=15)
                man_hour = st.slider("Hour (UTC)", min_value=0, max_value=23, value=20)
            with fc3:
                man_state = st.selectbox("US State", list(US_STATE_MAP.values()), index=list(US_STATE_MAP.values()).index('TX'))
                man_tz = st.selectbox("TimeZone", ['US/Central', 'US/Eastern', 'US/Mountain', 'US/Pacific'], index=0)

            man_submit = st.form_submit_button("🚀 Run Manual Prediction", use_container_width=True)

        if man_submit:
            man_features = pd.DataFrame([{
                'precipitation_in': float(man_precip_in),
                'latitude': float(man_lat),
                'longitude': float(man_lon),
                'month': int(man_month),
                'day': int(man_day),
                'hour': int(man_hour),
                'timezone': str(man_tz),
                'season': str(get_season(man_month)),
                'state': str(man_state)
            }])

            m_storm_pred = int(occ_model.predict(man_features)[0])
            m_storm_prob = float(occ_model.predict_proba(man_features)[0][1])
            m_risk_tier = determine_risk_tier(m_storm_prob)

            st.markdown("#### Manual Mode Prediction Result")
            if m_storm_pred == 1:
                st.error(f"⛈️ **STORM DETECTED** — Probability: **{m_storm_prob*100:.1f}%** | Risk: **{m_risk_tier}**")
            else:
                st.success(f"☀️ **NO STORM DETECTED** — Probability: **{m_storm_prob*100:.1f}%** | Risk: **{m_risk_tier}**")

            if sev_model is not None:
                m_sev = sev_model.predict(man_features)[0]
                st.info(f"Predicted Severity: **{m_sev}**")


if __name__ == "__main__":
    main()
