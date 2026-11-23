# ⛈️ StormCast: Live Weather-Based Storm Risk Prediction

StormCast is a machine learning application that connects live meteorological observations to trained Scikit-Learn pipelines to predict storm occurrence likelihood, risk tier, and hazard severity for any chosen location.

Trained on historical observations from the NOAA Automated Surface Observing System (ASOS) dataset (8.6+ million weather events across 2,071 weather stations), the application bridges historical pattern recognition with live atmospheric data.

---

## 🚀 Key Features

- **Live Meteorological Ingestion:** Fetches real-time temperature, precipitation, humidity, wind speed, and observation timestamp for user-specified cities.
- **Dual API Support:**
  - **Open-Meteo API:** Default live meteorological service with no API key or sign-up required.
  - **OpenWeatherMap API:** Supported when an optional API key is configured.
- **Trained Machine Learning Inference:**
  - **Storm Occurrence:** Evaluates spatial coordinates, diurnal cycle, and seasonal indicators using a calibrated Random Forest pipeline.
  - **Hazard Severity:** Classifies expected event intensity across `Light`, `Moderate`, `Heavy`, and `Severe` categories with probability distributions.
- **Manual Fallback Mode:** An interactive offline mode enabling manual parameter specification for hypothetical or testing scenarios.
- **Rapid Startup:** Loads only trained model artifacts and preprocessing pipelines without loading raw multi-gigabyte training files.

---

## 🏗️ System Architecture & Prediction Flow

```
User Selects Location (City / Region)
                 │
                 ▼
Live Weather API (Open-Meteo / OpenWeatherMap)
                 │
                 ▼
Current Weather Parameters (Temp, Humidity, Rain mm, Wind, Timestamp)
                 │
                 ▼
Feature Engineering & Unit Mapping (mm -> inches, UTC hour, season, state, tz)
                 │
                 ▼
Trained Scikit-Learn Pipelines (StandardScaler + OneHotEncoder + Random Forest)
                 │
                 ▼
Final Prediction Display (Storm vs. No Storm | Probability % | Risk Tier | Severity)
```

---

## ⚙️ Configuration & API Setup

### Default Mode (No Setup Required)
By default, StormCast uses Open-Meteo's open API. It functions immediately upon launching without requiring any registration or configuration.

### Optional: OpenWeatherMap Integration
To use OpenWeatherMap as your primary data provider:
1. Sign up for a free account at [OpenWeatherMap](https://openweathermap.org/api) and generate an API key.
2. Create a `.env` file in the project root (or set an environment variable):
   ```env
   OPENWEATHER_API_KEY=your_actual_api_key_here
   ```
3. Alternatively, configure Streamlit secrets in `.streamlit/secrets.toml`:
   ```toml
   OPENWEATHER_API_KEY = "your_actual_api_key_here"
   ```

*Note: All `.env` and `secrets.toml` files are excluded from Git via `.gitignore`.*

---

## 💻 Installation & Quick Start

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### 2. Clone the Repository & Install Dependencies
```powershell
git clone https://github.com/Yash-o-Verma-01/StormCast.git
cd StormCast

# Install required Python packages
pip install -r requirements.txt
```

### 3. Run the Streamlit Application
```powershell
streamlit run app.py
```
The application will open in your default browser at `http://localhost:8501`.

---

## 🧪 Demonstration & Testing Walkthrough

1. **Enter a Location:**
   - Type a city name such as `Dallas`, `Miami`, `Denver`, `Seattle`, or `Austin`.
   - Click **📍 Get Weather** to pull live conditions (temperature, precipitation, humidity, wind).
2. **Execute Prediction:**
   - Click **⛈️ Predict Storm Risk**.
   - Observe the real-time inference result:
     - Storm Occurrence status (`STORM DETECTED` or `NO STORM DETECTED`)
     - Calibrated storm probability percentage
     - Risk Level category (`LOW`, `MODERATE`, `HIGH`)
     - Predicted weather severity (`Light`, `Moderate`, `Heavy`, `Severe`) with class breakdown.
3. **Manual / Offline Testing:**
   - Expand the **⚙️ Manual Prediction Mode (Fallback / Offline)** card.
   - Adjust latitude, longitude, precipitation, month, and hour to test hypothetical scenarios.

---

## 🔍 Model Limitations & Scope

- **Statistical Pattern Classifier:** StormCast estimates storm probability based on statistical patterns learned from historical ASOS tabular records. It is an educational prototype and is not a replacement for official numerical weather prediction (NWP) models (e.g., HRRR, GFS).
- **Not an Emergency Alert System:** This system must not be used for operational aviation dispatch, public safety alerts, or life-safety emergency decisions.
- **Future Scope:** Integrating radar reflectivity grids, convective available potential energy (CAPE), and multi-hour forecasting lead times via recurrent/transformer sequence models.

---

## 👤 Author
**Tanvi**
**Yash Verma**  
Bennett University
