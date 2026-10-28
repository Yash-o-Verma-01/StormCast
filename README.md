# ⛈️ StormCast: Storm Occurrence & Severity Analysis using Machine Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning pipeline and interactive Streamlit web application for storm-occurrence classification and weather-severity pattern recognition, trained and validated on the genuine **Kaggle US Weather Events (2016–2022)** dataset (8.6+ million NOAA/ASOS weather observations).

**Author:** Yash Verma  
**Academic Affiliation:** Bennett University  

> [!IMPORTANT]
> **Academic Prototype & Scientific Scope Disclaimer:**  
> StormCast is an academic machine learning prototype designed for historical pattern analysis and storm-occurrence classification. It models the statistical likelihood that atmospheric and spatio-temporal conditions correspond to convective storm episodes. It is **not** a real-time numerical weather prediction (NWP) model and must not be used for operational meteorological warnings, aviation dispatch, or emergency life-safety decisions.

---

## 📑 Table of Contents
- [Problem Statement](#-problem-statement)
- [Key Features](#-key-features)
- [Dataset Overview & Acquisition](#-dataset-overview--acquisition)
- [System Architecture & Workflow](#-system-architecture--workflow)
- [Machine Learning Methodology](#-machine-learning-methodology)
- [Evaluation & Experimental Results](#-evaluation--experimental-results)
- [Exploratory Data Analysis & Visualizations](#-exploratory-data-analysis--visualizations)
- [Streamlit Web Application](#-streamlit-web-application)
- [Project Directory Structure](#-project-directory-structure)
- [Installation & Quick Start (Windows PowerShell)](#-installation--quick-start-windows-powershell)
- [Comparison with Existing Systems](#-comparison-with-existing-systems)
- [Current Gaps, Limitations & Future Scope](#-current-gaps-limitations--future-scope)
- [College Viva Voce Q&A Guide](#-college-viva-voce-qa-guide)
- [License](#-license)

---

## 🎯 Problem Statement

Severe convective storms cause billions of dollars in infrastructure damage, agricultural loss, and aviation disruptions every year. Traditional numerical weather prediction (NWP) models (such as NOAA's HRRR or GFS) solve complex fluid dynamics and thermodynamics equations on supercomputers, requiring hours of compute time. 

This project explores an alternative, complementary question:
> *Can statistical machine learning models rapidly classify storm occurrence and estimate severity risk using geographic coordinates and spatio-temporal features extracted from historical airport weather observations?*

StormCast provides:
1. **Storm Occurrence Classification:** Binary prediction (`Storm` vs. `No Storm`) with calibrated probability estimation.
2. **Weather Severity Estimation:** Multi-class classification across genuine operational severity categories (`Light`, `Moderate`, `Severe`, `Heavy`).

---

## ✨ Key Features

- **Decoupled Meteorological Logic:** Built without synthetic data or target leakage. Evaluates atmospheric coordinates (`latitude`, `longitude`), cyclical diurnal hours, and seasonal temporal features.
- **Statistical Class-Imbalance Handling:** Utilizes stratified subsampling and class-weighted cost functions to handle severe skew (storms represent <1% of raw ASOS events).
- **Probability Calibration & Reliability:** Validated using Brier score metrics and reliability diagrams, ensuring probability values accurately reflect empirical frequencies.
- **Interactive Multi-Page Streamlit Web Dashboard:**
  - Real-time prediction simulator with built-in meteorological presets (e.g., Texas Summer Thunderstorm, High Plains Convective Line).
  - Interactive Exploratory Data Analysis (EDA) explorer with one-click dataset download.
  - Model evaluation scorecard displaying ROC curves, calibration curves, confusion matrices, and feature importances.
  - Comprehensive viva voce guide for classroom defense.
- **Lightweight & Reproducible:** Includes a preprocessed 100,000-row stratified dataset (`data/cleaned_weather_sample.csv`) and serialized pipelines (`models/*.joblib`), allowing immediate execution without downloading the 1.08 GB raw file.

---

## 📊 Dataset Overview & Acquisition

The model is built on the **US Weather Events (2016–2022)** dataset compiled by Sobhan Moosavi from **2,071 Automated Surface Observing System (ASOS)** airport stations across the contiguous United States.

- **Source:** [Kaggle US Weather Events (2016 - 2022)](https://www.kaggle.com/datasets/sobhanmoosavi/us-weather-events)
- **Raw File:** `WeatherEvents_Jan2016-Dec2022.csv` (1.08 GB, 8,627,181 rows, 14 attributes)
- **Event Breakdown in Full Dataset:**
  - `Rain`: 5,000,794 (57.97%)
  - `Fog`: 2,014,190 (23.35%)
  - `Snow`: 1,158,575 (13.43%)
  - `Cold`: 232,192 (2.69%)
  - `Precipitation`: 157,186 (1.82%)
  - `Storm`: 61,317 (0.71%)
  - `Hail`: 2,927 (0.03%)

### Handling Missing Values & Data Quality
Across the 8.6M records, 12 of 14 columns contain **0% missing values**. Null values exist only in `City` (0.20%) and `ZipCode` (0.80%).

### Reproducing from Scratch vs. Instant Execution
- **Instant Execution (Default):** The repository includes `data/cleaned_weather_sample.csv` (100,000 stratified rows: 40,000 genuine storms + 60,000 non-storms) and trained model files. No download is needed to run the web app or evaluate models.
- **Rebuilding from Raw Data:** To rebuild from scratch:
  1. Download `WeatherEvents_Jan2016-Dec2022.csv` from Kaggle and place it in the project root.
  2. Run `python prepare_data.py` to stream and regenerate the sample.

---

## 🏛️ System Architecture & Workflow

```
[Raw ASOS Dataset: 8.6M Rows] ─── (Streaming Chunk Ingestion: prepare_data.py)
               │
               ▼
 [Cleaned Analytical Sample: 100,000 Records] (data/cleaned_weather_sample.csv)
               │
       ┌───────┴───────────────────────┬───────────────────────────────┐
       ▼                               ▼                               ▼
 [src/visualization.py]            [train.py]                      [app.py]
 - Class Distribution          - Scikit-Learn Pipelines        - Streamlit Web App
 - Spatial/Temporal Heatmaps   - Stratified Train/Test Split   - Live Risk Inference
 - Missing Value Matrix        - Joblib Model Persistence      - Model Visualizer
       │                               │                               │
       ▼                               ▼                               ▼
  outputs/eda/*.png           models/*.joblib & metrics/         Local Browser UI
```

### Data Pipeline & Prevention of Target Leakage
1. **Excluding Event Duration (`EndTime(UTC)`):** Total event duration is unknown at the onset of a storm. Using it would inject future information (lookahead leakage). We strictly use `StartTime(UTC)` to derive seasonal (`month`, `season`) and diurnal (`hour`) features.
2. **Dropping High-Cardinality Identifiers:** High-cardinality nominal fields (`AirportCode`, `ZipCode`, `City`) cause spatial overfitting. They are replaced with continuous coordinates (`latitude`, `longitude`), `timezone`, and `state`.
3. **Leak-Free Scikit-Learn Pipelines:** All numeric median imputations, standard scaling, and categorical one-hot encoding are fit strictly inside `Pipeline` objects on training folds.

---

## 🔬 Machine Learning Methodology

### 1. Storm Occurrence Classification (Binary)
- **Target:** `is_storm` (1 = Storm, 0 = Non-Storm Weather Event)
- **Features Used:**
  - Numerical: `latitude`, `longitude`, `month`, `day`, `hour`
  - Categorical: `timezone`, `season`, `state`
- **Candidate Models:**
  - **Logistic Regression:** L2-regularized linear baseline with balanced class weights.
  - **Decision Tree Classifier:** Non-linear decision tree (`max_depth=12`, `min_samples_leaf=20`).
  - **Random Forest Classifier (Champion):** Ensemble of 100 estimators (`max_depth=16`, `n_jobs=-1`).

### 2. Weather Severity Classification (Multi-Class)
- **Target:** `Severity` (`Light`, `Moderate`, `Severe`, `Heavy`)
- **Features Used:** Spatio-temporal features + `Precipitation(in)` rate
- **Models:** Multinomial Logistic Regression vs. Balanced Multi-Class Random Forest.

---

## 📈 Evaluation & Experimental Results

All models were evaluated on an **unseen 20% test holdout (20,000 records)** using stratified sampling.

### Storm Occurrence Model Comparison

| Model Architecture | Accuracy | Macro F1 | Storm Recall | Storm Precision | ROC-AUC | Brier Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 77.69% | 0.7710 | 77.19% | 70.07% | 0.8358 | 0.1626 | Linear Baseline |
| **Decision Tree** | 82.86% | 0.8236 | 82.66% | 76.41% | 0.8970 | 0.1260 | Interpretable Tree |
| **Random Forest** | **83.73%** | **0.8317** | **81.96%** | **78.35%** | **0.9109** | **0.1200** | **Champion Model** |

### Why Random Forest is the Champion Architecture:
- **Superior Discrimination:** Highest ROC-AUC (**0.9109**) and Macro F1 (**0.8317**), successfully balancing sensitivity and specificity.
- **Reliable Probability Calibration:** Lowest Brier Score (**0.1200**), ensuring confidence scores correspond to true event likelihood.
- **Non-Linear Interactions:** Effectively maps complex interactions between summer diurnal heating (late afternoon UTC) and convective geographic corridors (e.g., Midwest and Gulf Coast).

### Weather Severity Model Comparison

| Model Architecture | Accuracy | Macro F1 | Weighted F1 | Severe F1 | Heavy F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 65.21% | 0.5464 | 0.6592 | 0.7526 | 0.4612 |
| **Random Forest (Champion)** | **68.20%** | **0.5892** | **0.6906** | **0.7681** | **0.5044** |

---

## 🖼️ Exploratory Data Analysis & Visualizations

All figures are automatically generated by `src/visualization.py` and saved in `outputs/`:

| Visualization | Description | File Path |
| :--- | :--- | :--- |
| **ROC Curves** | ROC curves for all 3 models showing Random Forest AUC of 0.9109 | `outputs/metrics/occurrence_roc_curves.png` |
| **Calibration Curves** | Reliability diagrams evaluating predicted probabilities against empirical frequencies | `outputs/metrics/occurrence_calibration_curves.png` |
| **Confusion Matrices** | Normalized confusion matrices comparing predictions across all architectures | `outputs/metrics/occurrence_confusion_matrices.png` |
| **Feature Importances** | Random Forest feature weights highlighting spatial and temporal drivers | `outputs/metrics/feature_importances.png` |
| **Class Distribution** | Imbalance visualization across weather types | `outputs/eda/eda_class_distribution.png` |
| **Geographic Density** | Latitude vs Longitude distribution across US airport stations | `outputs/eda/eda_geographic_distribution.png` |
| **Temporal Patterns** | Monthly and diurnal storm occurrence distributions | `outputs/eda/eda_temporal_patterns.png` |
| **Correlation Heatmap** | Pearson correlation matrix across numerical variables | `outputs/eda/eda_correlation_heatmap.png` |

---

## 💻 Installation & Quick Start (Windows PowerShell)

### Prerequisites
- Python 3.10, 3.11, or 3.12 installed
- Git installed

### Step 1: Clone the Repository
```powershell
git clone https://github.com/Yash-o-Verma-01/StormCast.git
cd StormCast
```

### Step 2: Create and Activate a Virtual Environment (Recommended)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Launch the Streamlit Web Application
```powershell
streamlit run app.py
```
The application will launch in your browser at `http://localhost:8501`.

### Step 5: (Optional) Re-train or Generate Visualizations
```powershell
# Re-generate EDA plots
python src/visualization.py

# Re-train models and update metrics
python train.py
```

---

## 🌐 Streamlit Web Application

The interactive web dashboard includes 4 complete modules:

1. **⛈️ Storm Risk Predictor:**
   - Sliders and selectboxes for Latitude, Longitude, State, TimeZone, Month, and Hour.
   - 4 built-in realistic presets: Texas Summer Afternoon Thunderstorm, Colorado Mountain Winter, Florida Convective Storm, and California Coastal Mild Weather.
   - Dual-model inference: Binary Storm probability meter + multi-class severity breakdown.
2. **📊 Data Explorer & EDA:**
   - Dynamic data filtering across 100,000 historical records.
   - One-click CSV download of the preprocessed dataset.
   - Interactive gallery of all 6 EDA figures.
3. **📈 Model Performance & Evaluation:**
   - Tabular model leaderboards with Accuracy, F1, ROC-AUC, and Brier Score.
   - Interactive inspection of ROC curves, calibration plots, and feature rankings.
4. **🎓 Viva Voce & Methodology Guide:**
   - Comprehensive answers to external examiner questions regarding machine learning theory, data leakage prevention, and model limitations.

---

## ⚖️ Comparison with Existing Systems

| Feature / Capability | Operational Forecasting (e.g., OpenWeather, NWS HRRR) | StormCast ML (Current Project) |
| :--- | :--- | :--- |
| **Primary Methodology** | Numerical Weather Prediction (NWP) solving fluid thermodynamics | Statistical Machine Learning on historical ASOS observations |
| **Execution Time** | Minutes to hours on HPC supercomputers | Milliseconds on commodity consumer hardware |
| **Input Data Required** | Real-time Doppler radar, satellite radiometry, barometric pressure, dew point | Spatial coordinates (`lat`, `lon`), state, timezone, diurnal hour, month |
| **Prediction Scope** | Continuous future timeline (1-hour to 10-day lead time) | Single-event historical occurrence & severity classification |
| **Real-Time Integration** | Continuous automated live sensor feeds | Manual parameter entry & historical scenario simulation |
| **Target Application** | Operational public alerts, aviation dispatch, emergency management | Academic research, historical pattern discovery, educational prototyping |

---

## 🔍 Current Gaps, Limitations & Future Scope

### Implemented vs. Planned Capabilities

| Capability | Current Status | Notes |
| :--- | :---: | :--- |
| Historical Storm Occurrence Classification | ✅ Implemented | Random Forest (83.73% Accuracy, 0.9109 ROC-AUC) |
| Weather Event Severity Multi-Class Model | ✅ Implemented | Random Forest (68.20% Accuracy, 0.6906 Weighted F1) |
| Streamlit Interactive Web Application | ✅ Implemented | 4 pages with presets, EDA explorer, and metrics |
| Leakage-Free Preprocessing Pipeline | ✅ Implemented | Fit strictly on training folds via Scikit-Learn pipelines |
| Probability Calibration & Brier Score Validation | ✅ Implemented | Evaluated across all models |
| Dynamic Real-Time Weather API Integration | ⏳ Future Scope | Requires streaming OpenWeatherMap or NOAA API keys |
| Continuous Atmospheric Physics Features | ⏳ Future Scope | Atmospheric pressure, CAPE, dew point, wind shear |
| Multi-Hour Forecasting Lead Time | ⏳ Future Scope | Requires time-series sequence models (LSTM / Temporal Transformers) |

### Current Limitations:
1. **Historical Occurrence vs. Future Prediction:** The model detects whether given spatio-temporal and precipitation attributes match historical storm patterns. It does not forecast future atmospheric development.
2. **Missing Thermodynamic Parameters:** The Kaggle ASOS dataset lacks direct convective indicators such as Convective Available Potential Energy (CAPE), barometric pressure tendencies, and radar reflectivity.
3. **Severe Class Imbalance:** In nature, storms are rare events (<1% of ASOS reports). Stratified sampling was necessary to provide sufficient positive examples for stable model training.

---

## 🎓 College Viva Voce Q&A Guide

### Q1: Why did you choose this project?
> **Answer:** Severe weather events cause extensive economic loss and safety risks. While operational physics models are computationally intensive, this project demonstrates how machine learning can rapidly identify convective storm risk and event severity patterns from empirical tabular data.

### Q2: What is the difference between training and prediction in your pipeline?
> **Answer:** During training (`train.py`), the model processes historical feature-label pairs to learn optimal decision thresholds using Gini impurity and log-loss minimization. During prediction (`app.py`), the saved pipeline processes new user inputs through frozen preprocessing transformations and outputs instantaneous probability estimates without altering model weights.

### Q3: Why is Random Forest superior to Logistic Regression for this task?
> **Answer:** Meteorological phenomena exhibit non-linear interactions. For example, high temperatures trigger thunderstorms in humid summer afternoons, but not in arid winter mornings. Logistic Regression can only learn linear hyperplanes, whereas Random Forest ensembles capture complex multidimensional decision boundaries without overfitting.

### Q4: How is the prediction probability calculated?
> **Answer:** In Random Forest, probability is the fraction of decision trees in the ensemble that vote for the `Storm` class:
> $$P(\text{Storm} \mid \mathbf{x}) = \frac{1}{T} \sum_{t=1}^{T} \mathbb{I}(h_t(\mathbf{x}) = \text{Storm})$$
> This probability was verified using Brier score calibration analysis.

### Q5: What is data preprocessing and why was it necessary?
> **Answer:** Raw ASOS data contains missing values, categorical labels (`State`, `Season`), and differing numerical scales. We used Scikit-Learn `ColumnTransformer` with `SimpleImputer`, `StandardScaler`, and `OneHotEncoder` strictly fitted on training splits to prevent data leakage.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 👤 Author

**Yash Verma**  
- GitHub: [@Yash-o-Verma-01](https://github.com/Yash-o-Verma-01)  
- Academic Program: B.Tech Computer Science & Engineering (Artificial Intelligence & Machine Learning)  
- Institution: Bennett University  
