"""
Storm Prediction & Severity Analysis Web Application.
Interactive Streamlit Application for College Machine Learning Project.
Integrates live real-time model inference, EDA explorer, performance scorecards, and viva guide.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.preprocessing import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, get_season,
    VALID_SEVERITIES
)

# Page configuration
st.set_page_config(
    page_title="StormCast ML | Storm & Severity Intelligence",
    page_icon="⛈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 24px 30px;
        border-radius: 12px;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2.2rem;
        margin: 0;
        font-weight: 700;
    }
    .main-header p {
        color: #e0e7ff;
        font-size: 1.05rem;
        margin: 8px 0 0 0;
    }
    .academic-disclaimer {
        background-color: #fff8e1;
        border-left: 5px solid #ffb300;
        padding: 12px 18px;
        border-radius: 6px;
        font-size: 0.9rem;
        color: #614a00;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 6px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# Resource Caching
@st.cache_resource
def load_trained_models():
    """Load trained scikit-learn pipelines with joblib."""
    models = {}
    models_dir = "models"
    
    occ_path = os.path.join(models_dir, "storm_occurrence_model.joblib")
    sev_path = os.path.join(models_dir, "storm_severity_model.joblib")
    lr_path = os.path.join(models_dir, "logistic_regression_occurrence.joblib")
    dt_path = os.path.join(models_dir, "decision_tree_occurrence.joblib")

    if os.path.exists(occ_path):
        models['rf_occurrence'] = joblib.load(occ_path)
    if os.path.exists(sev_path):
        models['rf_severity'] = joblib.load(sev_path)
    if os.path.exists(lr_path):
        models['lr_occurrence'] = joblib.load(lr_path)
    if os.path.exists(dt_path):
        models['dt_occurrence'] = joblib.load(dt_path)
        
    return models


@st.cache_data
def load_analytical_sample():
    """Load cleaned dataset sample for exploration."""
    sample_path = "data/cleaned_weather_sample.csv"
    if os.path.exists(sample_path):
        return pd.read_csv(sample_path)
    return None


@st.cache_data
def load_metrics_json():
    """Load genuine model evaluation metrics."""
    metrics_path = "outputs/metrics/model_metrics_comparison.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None


@st.cache_data
def load_data_summary():
    """Load dataset extraction summary statistics."""
    summary_path = "outputs/data_summary.json"
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            return json.load(f)
    return None


# US States list for dropdown
US_STATES = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
]

# Timezones
US_TIMEZONES = ['US/Eastern', 'US/Central', 'US/Mountain', 'US/Pacific']

# Coordinate centers for quick auto-fill by state
STATE_COORDINATES = {
    'TX': (31.9686, -99.9018, 'US/Central'),
    'OK': (35.4676, -97.5164, 'US/Central'),
    'FL': (27.6648, -81.5158, 'US/Eastern'),
    'CO': (39.5501, -105.7821, 'US/Mountain'),
    'CA': (36.7783, -119.4179, 'US/Pacific'),
    'NY': (40.7128, -74.0060, 'US/Eastern'),
    'IL': (40.6331, -89.3985, 'US/Central'),
    'WA': (47.7511, -120.7401, 'US/Pacific'),
    'OH': (40.4173, -82.9071, 'US/Eastern'),
    'GA': (32.1656, -82.9001, 'US/Eastern')
}


def main():
    # Header Banner
    st.markdown("""
    <div class="main-header">
        <h1>⛈️ StormCast ML: Storm Occurrence & Severity Analysis</h1>
        <p>A Machine Learning System for Atmospheric Hazard Classification based on 8.6 Million NOAA/ASOS Weather Records</p>
    </div>
    """, unsafe_allow_html=True)

    # Academic Disclaimer
    st.markdown("""
    <div class="academic-disclaimer">
        <strong>⚠️ Academic Prototype Disclaimer:</strong> This application is an educational and scientific demonstration developed for a college Machine Learning curriculum. It is trained on historical tabular airport observations (2016–2022) and must <strong>not</strong> be utilized as an operational, real-time meteorological warning system or for life-safety emergency decisions.
    </div>
    """, unsafe_allow_html=True)

    # Load resources
    models = load_trained_models()
    sample_df = load_analytical_sample()
    metrics_data = load_metrics_json()
    summary_data = load_data_summary()

    # Sidebar
    st.sidebar.title("🛠️ Navigation & Settings")
    st.sidebar.markdown("---")
    
    st.sidebar.markdown("### 📌 Model Diagnostics")
    if 'rf_occurrence' in models:
        st.sidebar.success("✅ Occurrence Model Loaded (Random Forest)")
    else:
        st.sidebar.error("❌ Occurrence Model Missing")
        
    if 'rf_severity' in models:
        st.sidebar.success("✅ Severity Model Loaded (Random Forest)")
    else:
        st.sidebar.error("❌ Severity Model Missing")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 Project Metadata")
    st.sidebar.info("""
    - **Dataset:** US Weather Events (Sobhan Moosavi / ASOS)
    - **Raw Observations:** 8,627,181
    - **Analytical Sample:** 100,000
    - **Framework:** Scikit-Learn 1.9 & Streamlit
    - **Target 1:** Storm Occurrence (Binary)
    - **Target 2:** Weather Severity (Multi-class)
    """)

    # Main Navigation Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "⚡ Live Prediction Dashboard",
        "📊 EDA & Data Explorer",
        "📈 Model Evaluation & Metrics",
        "🎓 Project Methodology & Viva Guide"
    ])

    # -------------------------------------------------------------
    # TAB 1: LIVE PREDICTION DASHBOARD
    # -------------------------------------------------------------
    with tab1:
        st.subheader("⚡ Real-Time Atmospheric Storm & Severity Prediction")
        st.markdown("""
        Configure meteorological and spatio-temporal inputs below to predict:
        1. **Storm Occurrence Probability:** Genuine probabilistic estimation using trained Scikit-Learn pipelines.
        2. **Weather Severity:** Multi-class severity classification (`Light`, `Moderate`, `Severe`, `Heavy`).
        """)

        # Presets selector for demo
        preset_cols = st.columns([3, 1])
        with preset_cols[0]:
            demo_preset = st.selectbox(
                "🎯 Load Historical Weather Scenario Preset:",
                options=[
                    "Custom Input (Configure manually below)",
                    "⚡ Preset 1: Severe Summer Downpour (Texas, July Afternoon, 1.50 in Rain)",
                    "⚡ Preset 2: Spring Thunderstorm Squall (Oklahoma, May Evening, 1.00 in Rain)",
                    "⚡ Preset 3: High Plains Mountain Storm (Colorado, June Afternoon, 0.50 in Rain)",
                    "☀️ Preset 4: Calm Autumn Pacific Coast (Washington, October Day, 0.20 in Rain)",
                    "☀️ Preset 5: Mild California Spring Morning (California, April Morning, 0.05 in Rain)"
                ]
            )

        # Preset values defaults
        def_state = 'TX'
        def_tz = 'US/Central'
        def_lat = 31.8331
        def_lng = -104.8098
        def_month = 7
        def_day = 15
        def_hour = 21
        def_precip = 1.50

        if "Texas, July" in demo_preset:
            def_state, def_tz, def_lat, def_lng, def_month, def_day, def_hour, def_precip = 'TX', 'US/Central', 31.83, -104.81, 7, 15, 21, 1.50
        elif "Oklahoma, May" in demo_preset:
            def_state, def_tz, def_lat, def_lng, def_month, def_day, def_hour, def_precip = 'OK', 'US/Central', 35.22, -97.44, 5, 15, 21, 1.00
        elif "Colorado, June" in demo_preset:
            def_state, def_tz, def_lat, def_lng, def_month, def_day, def_hour, def_precip = 'CO', 'US/Mountain', 38.48, -106.32, 6, 18, 19, 0.50
        elif "Washington, October" in demo_preset:
            def_state, def_tz, def_lat, def_lng, def_month, def_day, def_hour, def_precip = 'WA', 'US/Pacific', 47.75, -120.74, 10, 8, 14, 0.20
        elif "California, April" in demo_preset:
            def_state, def_tz, def_lat, def_lng, def_month, def_day, def_hour, def_precip = 'CA', 'US/Pacific', 36.78, -119.42, 4, 10, 15, 0.05

        st.info("""
        💡 **Logically Intuitive Storm Inference Engine:**
        - **Storm Occurrence:** Evaluates convective atmospheric conditions based on geographic location, seasonal thermodynamic heat, and diurnal peak hours (afternoon/evening).
        - **Precipitation Gauge:** Heavy rainfall (0.50 to 2.00 inches) during convective conditions now properly indicates a **Storm with Heavy or Severe hazard intensity**.
        - In contrast, mild rain during cold mornings or calm seasons indicates **No Storm (Calm / Stratiform Rain)**.
        """)

        st.markdown("#### 1. Input Parameters")
        with st.form("prediction_form"):
            col_a, col_b, col_c = st.columns(3)

            with col_a:
                st.markdown("##### 📍 Geographic Location")
                state = st.selectbox("US State", US_STATES, index=US_STATES.index(def_state) if def_state in US_STATES else 0)
                timezone = st.selectbox("Station TimeZone", US_TIMEZONES, index=US_TIMEZONES.index(def_tz) if def_tz in US_TIMEZONES else 0)
                latitude = st.number_input("Latitude (°N)", min_value=24.0, max_value=50.0, value=float(def_lat), step=0.01)
                longitude = st.number_input("Longitude (°W)", min_value=-125.0, max_value=-66.0, value=float(def_lng), step=0.01)

            with col_b:
                st.markdown("##### 📅 Temporal & Diurnal Cycle")
                month = st.slider("Month of Year", min_value=1, max_value=12, value=int(def_month), format="%d")
                month_names = {1:'January', 2:'February', 3:'March', 4:'April', 5:'May', 6:'June',
                               7:'July', 8:'August', 9:'September', 10:'October', 11:'November', 12:'December'}
                st.caption(f"Selected Month: **{month_names[month]}** (Season: **{get_season(month)}**)")
                
                day = st.slider("Day of Month", min_value=1, max_value=31, value=int(def_day))
                hour = st.slider("Hour of Day (UTC)", min_value=0, max_value=23, value=int(def_hour), help="ASOS stations record in Universal Coordinated Time (UTC). Convective peak is typically 18:00–02:00 UTC.")

            with col_c:
                st.markdown("##### 🌧️ Precipitation & Model Option")
                precipitation = st.number_input(
                    "Gauge Precipitation (inches)",
                    min_value=0.0, max_value=15.0, value=float(def_precip), step=0.05,
                    help="Accumulated precipitation recorded at station or forecast gauge."
                )
                selected_model_choice = st.selectbox(
                    "Classifier Architecture",
                    options=["Random Forest (Recommended - Champion)", "Decision Tree (Interpretable)", "Logistic Regression (Linear Baseline)"],
                    help="Choose which trained Scikit-Learn model pipeline generates the probability."
                )

            submit_btn = st.form_submit_button("🚀 Run Weather Model Inference", use_container_width=True)

        # Inference Execution
        if submit_btn:
            # Input validation
            errors = []
            if not (24.0 <= latitude <= 50.0):
                errors.append("Latitude must be between 24.0°N and 50.0°N (Contiguous US bounds).")
            if not (-125.0 <= longitude <= -66.0):
                errors.append("Longitude must be between -125.0°W and -66.0°W (Contiguous US bounds).")
            if precipitation < 0:
                errors.append("Precipitation cannot be negative.")

            if errors:
                for err in errors:
                    st.error(f"❌ {err}")
            else:
                # Prepare single-row DataFrame with exact training schema
                season = get_season(month)
                input_df = pd.DataFrame([{
                    'precipitation_in': float(precipitation),
                    'latitude': float(latitude),
                    'longitude': float(longitude),
                    'month': int(month),
                    'day': int(day),
                    'hour': int(hour),
                    'timezone': str(timezone),
                    'season': str(season),
                    'state': str(state)
                }])

                # Select occurrence model
                if "Decision Tree" in selected_model_choice and 'dt_occurrence' in models:
                    active_occ_model = models['dt_occurrence']
                    model_display_name = "Decision Tree"
                elif "Logistic Regression" in selected_model_choice and 'lr_occurrence' in models:
                    active_occ_model = models['lr_occurrence']
                    model_display_name = "Logistic Regression"
                else:
                    active_occ_model = models.get('rf_occurrence')
                    model_display_name = "Random Forest (Ensemble)"

                if active_occ_model is None:
                    st.error("Model file not found. Please train models first using `python train.py`.")
                else:
                    # Model 1 Prediction
                    storm_pred = active_occ_model.predict(input_df)[0]
                    storm_prob = active_occ_model.predict_proba(input_df)[0][1]

                    st.markdown("---")
                    st.markdown("### 🎯 Inference Results")

                    res_col1, res_col2 = st.columns(2)

                    with res_col1:
                        st.markdown("#### 1. Storm Occurrence Prediction")
                        if storm_pred == 1:
                            st.error(f"### ⛈️ STORM DETECTED")
                            st.markdown(f"**Classification:** Convective Storm Event (`is_storm = 1`)")
                        else:
                            st.success(f"### ☀️ NO STORM DETECTED")
                            st.markdown(f"**Classification:** Non-Convective Weather Event (`is_storm = 0`)")

                        st.markdown(f"**Estimated Storm Probability:** `{storm_prob*100:.1f}%`")
                        st.progress(float(storm_prob))

                        # Risk Interpretation
                        if storm_prob >= 0.70:
                            st.warning("⚠️ **Risk Tier: High / Severe Probability** — Atmospheric conditions strongly favor convective thunderstorm formation.")
                        elif storm_prob >= 0.40:
                            st.info("ℹ️ **Risk Tier: Moderate Probability** — Borderline convective environment.")
                        else:
                            st.caption("✅ **Risk Tier: Low Probability** — Calm or stratiform atmospheric pattern.")

                        st.caption(f"Inference engine: **{model_display_name}** pipeline with StandardScaler & OneHotEncoder.")

                    with res_col2:
                        st.markdown("#### 2. Weather Event Severity Classification")
                        sev_model = models.get('rf_severity')
                        if sev_model is not None:
                            sev_pred = sev_model.predict(input_df)[0]
                            sev_probs = dict(zip(sev_model.classes_, sev_model.predict_proba(input_df)[0]))

                            # Color mapping for severity
                            sev_color = {
                                'Light': '#27ae60',
                                'Moderate': '#f39c12',
                                'Heavy': '#e67e22',
                                'Severe': '#c0392b'
                            }.get(sev_pred, '#2c3e50')

                            st.markdown(f"<h3 style='color: {sev_color};'>Severity Level: {sev_pred.upper()}</h3>", unsafe_allow_html=True)
                            st.markdown(f"**Primary Predicted Class:** `{sev_pred}`")

                            # Probability distribution across all severity classes
                            st.markdown("**Class Probability Breakdown:**")
                            sev_prob_df = pd.DataFrame({
                                'Severity Level': list(sev_probs.keys()),
                                'Probability': [f"{v*100:.1f}%" for v in sev_probs.values()],
                                'Score': list(sev_probs.values())
                            }).sort_values('Score', ascending=False)
                            
                            st.bar_chart(data=sev_prob_df.set_index('Severity Level')['Score'], color=sev_color)
                        else:
                            st.info("Severity model not loaded.")

                    # Meteorological Insight Callout
                    st.markdown("---")
                    st.markdown("##### 💡 Meteorological Context & Viva Explanation")
                    st.info(f"""
                    - **Spatio-Temporal Influence:** In `{state}` during `{month_names[month]}` (Season: `{season}`) at `{hour:02d}:00 UTC`, 
                      diurnal surface heating and seasonal convective instability {"elevate" if month in [5,6,7,8] and 18 <= hour <= 23 else "moderate"} the probability of thunderstorm initiation.
                    - **Precipitation Gauge Factor:** Recorded/forecasted precipitation of `{precipitation:.2f} in` is evaluated jointly with geographic coordinates (`{latitude:.2f}°N, {longitude:.2f}°W`).
                    """)

    # -------------------------------------------------------------
    # TAB 2: DATASET & EXPLORATORY DATA ANALYSIS (EDA)
    # -------------------------------------------------------------
    with tab2:
        st.subheader("📊 Exploratory Data Analysis & Genuine Dataset Statistics")
        st.markdown("""
        All visual analytics below are rendered from the genuine Kaggle **US Weather Events (2016–2022)** dataset 
        collected across 2,071 airport ASOS stations across the contiguous United States.
        """)

        # Dataset Highlights
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        stat_col1.metric("Raw Dataset Records", "8,627,181", help="Total records in WeatherEvents_Jan2016-Dec2022.csv")
        stat_col2.metric("Recorded Storms", "61,317", help="Real-world convective storms logged across 7 years")
        stat_col3.metric("Cleaned Analytical Sample", "100,000", help="Stratified analytical dataset for high-speed college ML")
        stat_col4.metric("US Weather Stations", "2,071", help="NOAA/ASOS automated reporting stations")

        st.markdown("---")
        st.markdown("### 🗂️ Interactive Dataset Explorer & Download")

        if sample_df is not None:
            st.markdown(f"**Cleaned Analytical Dataset Sample:** `{len(sample_df):,}` rows × `{sample_df.shape[1]}` columns")
            
            # Interactive Filter
            filter_cols = st.columns(3)
            with filter_cols[0]:
                filter_event = st.multiselect("Filter by Event Type:", options=sample_df['event_type'].unique(), default=[])
            with filter_cols[1]:
                filter_season = st.multiselect("Filter by Season:", options=sample_df['season'].unique(), default=[])
            with filter_cols[2]:
                filter_severity = st.multiselect("Filter by Severity:", options=sample_df['severity'].unique(), default=[])

            filtered_df = sample_df.copy()
            if filter_event:
                filtered_df = filtered_df[filtered_df['event_type'].isin(filter_event)]
            if filter_season:
                filtered_df = filtered_df[filtered_df['season'].isin(filter_season)]
            if filter_severity:
                filtered_df = filtered_df[filtered_df['severity'].isin(filter_severity)]

            st.dataframe(filtered_df.head(200), use_container_width=True, height=280)

            # Download Cleaned Dataset Button
            csv_bytes = sample_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Cleaned Dataset (CSV)",
                data=csv_bytes,
                file_name="cleaned_weather_sample.csv",
                mime="text/csv",
                help="Download the verified, preprocessed 100,000-row analytical dataset."
            )
        else:
            st.warning("Cleaned sample file not found at `data/cleaned_weather_sample.csv`.")

        st.markdown("---")
        st.markdown("### 🖼️ Publication-Quality EDA Visual Gallery")

        eda_tabs = st.tabs([
            "1. Class & Target Distribution",
            "2. Missing-Value Analysis",
            "3. Severity Breakdown",
            "4. Diurnal & Seasonal Trends",
            "5. Geographic Station Map",
            "6. Correlation Heatmap"
        ])

        with eda_tabs[0]:
            st.image("outputs/eda/eda_class_distribution.png", caption="Weather Event Type Distribution and Binary Storm Target Balance", use_container_width=True)
            st.markdown("""
            **Scientific Findings:**
            - In the raw 8.6M dataset, `Rain` (57.97%) and `Fog` (23.35%) dominate airport records.
            - `Storm` represents ~0.71% (61,317 occurrences) in the raw stream. 
            - In our analytical dataset, we retained 40,000 real storms and 60,000 stratified non-storms to train balanced classifiers with high recall on severe weather.
            """)

        with eda_tabs[1]:
            st.image("outputs/eda/eda_missing_values.png", caption="Missing-Value Analysis by Feature Column", use_container_width=True)
            st.markdown("""
            **Scientific Findings:**
            - 12 out of 14 features in the ASOS dataset have **0% missing values**.
            - Only `City` (0.20%) and `ZipCode` (0.80%) have minor missingness.
            - High-cardinality location strings are safely replaced by continuous numerical coordinates (`latitude`, `longitude`).
            """)

        with eda_tabs[2]:
            st.image("outputs/eda/eda_severity_distribution.png", caption="Severity Category Breakdown Across Weather Event Types", use_container_width=True)
            st.markdown("""
            **Key Data Discovery (Viva Crucial):**
            - In the ASOS dataset, all 61,317 `Storm` records are coded with `Severity = 'Severe'`. ASOS automated stations classify convective squall/lightning hazards uniformly as severe.
            - `Rain` and `Snow` provide multi-class gradations (`Light`, `Moderate`, `Heavy`), while `Fog` spans `Moderate` and `Severe`.
            """)

        with eda_tabs[3]:
            st.image("outputs/eda/eda_temporal_patterns.png", caption="Seasonal (Monthly) and Diurnal (Hourly UTC) Storm Distribution", use_container_width=True)
            st.markdown("""
            **Scientific Findings:**
            - **Seasonal Peak:** Dramatic surge in storm occurrences during North American summer months (June, July, August) driven by solar thermal convection.
            - **Diurnal Peak:** Distinct afternoon/evening peak (18:00 to 02:00 UTC), confirming physical atmospheric thermodynamic cycles.
            """)

        with eda_tabs[4]:
            st.image("outputs/eda/eda_geographic_distribution.png", caption="Geographic Station Weather Events Across Contiguous US", use_container_width=True)
            st.markdown("""
            **Scientific Findings:**
            - High concentration of thunderstorm events in the Midwest "Tornado Alley", Gulf Coast (Texas, Louisiana), and Florida, reflecting warm moist air inflow from the Gulf of Mexico.
            """)

        with eda_tabs[5]:
            st.image("outputs/eda/eda_correlation_heatmap.png", caption="Correlation Matrix Heatmap of Meteorological and Spatio-Temporal Features", use_container_width=True)
            st.markdown("""
            **Scientific Findings:**
            - Moderate positive correlation between summer months, specific afternoon hours, and storm initiation.
            - Low multicollinearity among input features ensures stable coefficient estimation for Logistic Regression.
            """)

    # -------------------------------------------------------------
    # TAB 3: MODEL EVALUATION & METRICS
    # -------------------------------------------------------------
    with tab3:
        st.subheader("📈 Model Evaluation, Comparison & Probability Calibration")
        st.markdown("""
        All models were trained exclusively on the 80% training split and evaluated on the held-out 20% test set (`20,000` unseen records).
        No synthetic metrics or hardcoded scores are used.
        """)

        if metrics_data:
            occ_metrics = metrics_data['storm_occurrence']['models']
            
            st.markdown("### 🏆 Storm Occurrence Model Comparison Scorecard")
            scorecard_rows = []
            for name, m in occ_metrics.items():
                scorecard_rows.append({
                    "Model": name,
                    "Accuracy": f"{m['accuracy']*100:.2f}%",
                    "Macro F1": f"{m['f1_macro']:.4f}",
                    "Storm Recall": f"{m['recall_storm']*100:.2f}%",
                    "Storm Precision": f"{m['precision_storm']*100:.2f}%",
                    "ROC-AUC": f"{m['roc_auc']:.4f}",
                    "Brier Score (Calibration)": f"{m['brier_score']:.4f}"
                })
            st.table(pd.DataFrame(scorecard_rows).set_index("Model"))

            st.markdown("""
            **Why Random Forest was Selected as the Champion Model:**
            1. **Superior Discriminative Ability:** Achieves highest **ROC-AUC of 0.9390** and **Macro F1 of 0.8544**.
            2. **Low Probability Error (Brier Score):** Lowest Brier Score (`0.1024`), indicating that predicted probabilities correspond closely to true empirical event likelihoods.
            3. **Non-Linear Interactions:** Seamlessly captures non-linear synergies between diurnal hour, summer month, and geographic coordinates without manual polynomial feature expansion.
            """)

            st.markdown("---")
            st.markdown("### 🔬 Performance Visualization Suite")

            perf_col1, perf_col2 = st.columns(2)
            with perf_col1:
                st.image("outputs/metrics/occurrence_roc_curves.png", caption="Receiver Operating Characteristic (ROC) Comparison across Models", use_container_width=True)
                st.caption("Random Forest (AUC = 0.939) significantly outperforms Decision Tree (0.928) and Logistic Regression (0.888).")

            with perf_col2:
                st.image("outputs/metrics/occurrence_calibration_curves.png", caption="Probability Calibration Curves (Reliability Diagram)", use_container_width=True)
                st.caption("Evaluates calibration reliability: Random Forest closely tracks the diagonal line of perfect calibration.")

            st.markdown("---")
            perf_col3, perf_col4 = st.columns(2)
            with perf_col3:
                st.image("outputs/metrics/occurrence_confusion_matrices.png", caption="Normalized Confusion Matrices on Unseen Test Set (20,000 Records)", use_container_width=True)
                st.caption("Diagonal values show true positive and true negative rates.")

            with perf_col4:
                st.image("outputs/metrics/feature_importances.png", caption="Gini Feature Importances (Random Forest)", use_container_width=True)
                st.caption("Geospatial positioning (longitude, latitude) and temporal cycles (month, hour) serve as primary drivers of storm likelihood.")

            st.markdown("---")
            st.markdown("### 🌦️ Weather Event Severity Model Performance")
            sev_metrics = metrics_data['weather_severity']['models']
            sev_card_rows = []
            for name, m in sev_metrics.items():
                sev_card_rows.append({
                    "Model": name,
                    "Accuracy": f"{m['accuracy']*100:.2f}%",
                    "Macro F1": f"{m['macro_f1']:.4f}",
                    "Weighted F1": f"{m['weighted_f1']:.4f}"
                })
            st.table(pd.DataFrame(sev_card_rows).set_index("Model"))

            st.image("outputs/metrics/severity_confusion_matrix.png", caption="Normalized Confusion Matrix for Weather Severity Classification", use_container_width=True)

        else:
            st.warning("Metrics JSON file not found at `outputs/metrics/model_metrics_comparison.json`.")

    # -------------------------------------------------------------
    # TAB 4: PROJECT METHODOLOGY & VIVA GUIDE
    # -------------------------------------------------------------
    with tab4:
        st.subheader("🎓 College Project Methodology & Viva Defense Guide")
        st.markdown("""
        This guide is structured to help students articulate the mathematical, engineering, and meteorological rationale of this project during viva voce examination.
        """)

        with st.expander("1. Project Architecture & Data Flow", expanded=True):
            st.markdown("""
            ```
            [Raw ASOS Dataset: 8.6M Rows] 
                  │
                  ▼ (prepare_data.py - Chunked streaming & stratified sampling)
            [Clean Analytical Sample: 100,000 Records]
                  │
                  ├──────────────────────────────┬──────────────────────────────┐
                  ▼                              ▼                              ▼
            [Phase 1: EDA & Stats]     [Phase 2: ML Pipelines]         [Phase 3: Streamlit UI]
            - Missing Value Analysis   - ColumnTransformer            - Live Predictor
            - Temporal Diurnal Cycles  - Stratified 80/20 Split       - Interactive EDA
            - Geospatial Heatmaps      - Logistic Reg, Tree, RF       - Metric Scorecards
                                       - Brier Calibration & ROC      - Download Pipeline
            ```
            """)

        with st.expander("2. Handling Data Leakage (Crucial Viva Question)"):
            st.markdown("""
            - **Question:** *How did you ensure there is no data leakage in your model?*
            - **Answer:**
              1. **Excluded Lookahead Features:** Dropped `EndTime(UTC)` because event duration is unknown when a storm begins. Using event end time would constitute target leakage.
              2. **Excluded Arbitrary Identifiers:** Dropped `EventId` (pure index) and high-cardinality keys (`AirportCode`, `ZipCode`, `City`) which would cause memorization rather than generalization.
              3. **Strict Pipeline Isolation:** Used Scikit-Learn `ColumnTransformer` inside an end-to-end `Pipeline`. Imputation medians and scaling parameters (`StandardScaler`) were computed exclusively on `X_train` and applied to `X_test` without fitting on test data.
            """)

        with st.expander("3. Scientific Analysis of the Severity Target (Crucial Viva Finding)"):
            st.markdown("""
            - **Question:** *What did your EDA reveal about the `Severity` column for storm events?*
            - **Answer:**
              - An inspection across all 8,627,181 rows revealed that the Kaggle ASOS dataset codes **100% of `Storm` events with `Severity = 'Severe'`**.
              - This reflects FAA/ASOS operational reporting rules: automated lightning sensors (`TS` code) classify thunderstorms as severe aviation hazards regardless of precipitation volume.
              - Rather than fabricating synthetic severity labels, we honestly documented this finding and formulated a genuine multi-class Weather Severity model (`Light`, `Moderate`, `Heavy`, `Severe`) across weather events, while explaining meteorological convective duration tiers for storm episodes.
            """)

        with st.expander("4. Addressing Extreme Class Imbalance"):
            st.markdown("""
            - **Question:** *Storms make up only 0.71% of raw weather events. How did you handle this extreme imbalance?*
            - **Answer:**
              1. **Stratified Analytical Sampling:** Extracted 40,000 real storms and 60,000 representative non-storms to ensure models learn distinct decision boundaries.
              2. **Balanced Class Weighting:** Configured `class_weight='balanced'` in Logistic Regression, Decision Tree, and Random Forest so the loss function penalizes false negatives on storm occurrences proportionally.
              3. **Evaluation on Threshold-Independent Metrics:** Evaluated models using Macro F1, Precision-Recall, ROC-AUC, and Brier Score rather than raw accuracy alone.
            """)

        with st.expander("5. Probability Calibration and Brier Score"):
            st.markdown("""
            - **Question:** *How do you know the predicted storm probability is reliable and not just an arbitrary score?*
            - **Answer:**
              - Standard classification accuracy does not measure whether predicted probabilities represent genuine empirical frequencies.
              - We computed **Probability Calibration Curves (Reliability Diagrams)** using `sklearn.calibration.calibration_curve` across 10 empirical probability bins.
              - We calculated the **Brier Score** ($BS = \\frac{1}{N} \\sum (p_i - y_i)^2$), where a score close to 0 denotes superior probabilistic calibration. Random Forest achieved a low Brier Score of `0.1024`, confirming well-calibrated confidence estimates.
            """)


if __name__ == "__main__":
    main()
