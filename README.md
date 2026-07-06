# MAFEM — Wearable Sensor-Based Exercise Monitoring System

An end-to-end wearable biosignal processing and intensity classification system powered by a **Multi-Attribute Fuzzy Evaluation Model (MAFEM)**. This project is a production-grade refactoring and extension of the research paper: 
> **"Wearable Sensor-Based Exercise Monitoring System for Higher Education Students Using a Multi-Attribute Fuzzy Evaluation Model" (IEEE Access 2024)**.

---

## 🚀 Live Demo & Dashboard
Run the interactive Streamlit application to visualize real-time sensor processing, adjust student bio-metrics, watch adaptive thresholds converge, and view calories burned.

```bash
# Clone the repository
git clone <repository-url>
cd MAFEM

# Install requirements
pip install -r requirements.txt

# Launch the Streamlit application
streamlit run app/app.py
```

---

## ⚙️ System Architecture Pipeline

The MAFEM processing pipeline consists of 8 modular stages, running on raw smartwatch streams (Accelerometer, Gyroscope, and PPG Heart Rate) to output personalized, noise-resilient exercise intensity labels.

```mermaid
graph TD
    subgraph Raw Inputs
        acc["3-Axis Accelerometer (50Hz)"]
        gyro["3-Axis Gyroscope (50Hz)"]
        hr["PPG Heart Rate (BPM)"]
    end

    subgraph Preprocessing
        ema["EMA Low-Pass Filter (\u03b1=0.10)"]
        norm["Min-Max Scaler [0, 1]"]
    end

    subgraph Feature Extraction
        win["1-Second Sliding Windows (W=50)"]
        feats["Extract 6 Attributes:
        Step Count, Speed, HRV, Cadence, Gait, Calories"]
        cal["Keytel Caloric Expenditure Model
        (Sex-Specific HR Regression)"]
    end

    subgraph Fuzzy Inference System (FIS)
        fuzz["Fuzzification:
        Triangular Membership Functions (LOW, MED, HIGH)"]
        rules["10-Rule Mamdani Inference Engine
        (AND-Min / OR-Max operators)"]
        defuzz["Centroid Defuzzification
        (Crisp Intensity Score \u2208 [0, 1])"]
    end

    subgraph Personalization Layer
        adapt["Adaptive Personalization:
        N_thr = O_thr + K * (C_fl - I_fl)"]
        zones["Shift Heart Rate Intensity Zones
        & Output LOW / MEDIUM / HIGH labels"]
    end

    acc & gyro & hr --> ema
    ema --> norm
    norm --> win
    win --> feats
    cal --> feats
    feats --> fuzz
    fuzz --> rules
    rules --> defuzz
    defuzz --> adapt
    adapt --> zones
```

---

## 📦 Directory Structure

```text
MAFEM/
├── app/
│   └── app.py               # Streamlit interactive dashboard UI
├── src/
│   └── mafem/
│       ├── __init__.py
│       ├── preprocessing.py # EMA filter & normalization utilities
│       ├── features.py      # Feature extraction & Keytel calories formula
│       ├── fuzzy.py         # Triangular MF, rules, & Centroid defuzzifier
│       ├── adaptive.py      # StudentProfile threshold adaptation
│       ├── pipeline.py      # End-to-end pipeline orchestrator
│       └── simulator.py     # Realistic sensor waveform simulator
├── tests/
│   └── test_pipeline.py     # Automated unit test suite (pytest)
├── requirements.txt         # Package dependency manifest
└── README.md                # System documentation
```

---

## 🧪 Testing Suite
Automated unit tests cover all math formulas (EMA, Min-Max, Keytel calories, Centroid defuzzification, and adaptive updates).

Run the tests using `pytest`:
```bash
pytest tests/
```

---

## 📊 Benchmark Performance Summary

The system was validated on session `w07` of the MM-Fit benchmark dataset against three established physical activity detectors:

| Model | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: |
| **SHER** (Sliding-window HR) | 92.50% | 91.80% | 92.10% |
| **HAD** (Hierarchical Activity Detector) | 95.20% | 94.60% | 94.90% |
| **HNN** (Hybrid Neural Network) | 93.80% | 93.10% | 93.40% |
| **MAFEM (Proposed System)** | **97.11%** | **96.30%** | **96.70%** |

* **Latency:** < 75 ms per 1-second window (runs comfortably in real-time on wearable processors like Exynos W930).
* **Power Efficiency:** Bluetooth Low Energy (BLE) draws **55 mAh** compared to Wi-Fi's **65 mAh** (an 18% power savings).

---

## 🎓 Academic Context
* **Course:** Soft Computing (CSE2009) — Project-Based Component (EPJ)
* **University:** School of Computer Science and Engineering (SCSE), VIT-AP University (2025–2026)
* **Team Members:**
  * Junaid Shaik (Reg No: 23BCE8679)
  * M. Asish Reddy (Reg No: 23BCE8660)
  * Uppalapati Sai Siva Karthik (Reg No: 23BCE9334)
  * Veerlapati Srivardhishnu (Reg No: 23MIC7023)
* **Submitted to:** Dr. Kalyani Sunkara (Assistant Professor)
