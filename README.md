# Wind Turbine Active Power Prediction: A Benchmarking Study on the ENGIE Dataset

An end-to-end predictive modeling and benchmarking study to estimate wind turbine active power production (`TARGET`) using secondary physical SCADA sensor streams, intentionally excluding direct wind speed measurements.

**[Project Notebook](./notebooks/wind-turbine-analysis.ipynb)**
---

## Context & Physical Motivation
Under standard operating conditions, wind turbine active power output is modeled directly using wind speed measurements. However, physical anemometers on turbine nacelles frequently experience mechanical failure, sensor drift, or wake-effect turbulences from neighboring turbines, leading to unreliable readings.

This study implements and benchmarks an alternative modeling paradigm: **predicting active power output using only secondary physical and mechanical indicators** (shaft speeds, components temperatures, and pitch angles). 

---

## Data Pipeline & Preprocessing
To transition this study from exploratory analysis into a robust machine learning system, the following pipeline was established:
1. **Defensive Data Ingestion:** Implemented a dual-path loader that dynamically identifies local CSV sources first, falling back to secure `kagglehub` downloads if needed.
2. **Feature Selection & Pruning:** Programmatically removed high-missingness sensors (dropping four `Grid_voltage` columns missing $16.41\%$ of values) and resolved severe multicollinearity by dropping duplicate uncalibrated angles and redundant generator speed metrics.
3. **Relevance Filtering:** Applied an absolute Pearson correlation filter with a threshold of $0.03$ to eliminate weak predictive variables and statistical noise.
4. **Outlier-Robust Imputation:** Resolved remaining missing data (such as gearbox temperatures) using scikit-learn's `SimpleImputer` with a **median strategy**, protecting the distributions from unphysical SCADA sensor spikes.
5. **Chronological Splitting (Leakage Prevention):** Built a value-based $80/20$ chronological split utilizing the $80^{\text{th}}$ percentile of `Date_time`, ensuring duplicate timestamps across different turbines are kept together, establishing a strict validation boundary with zero temporal data leakage.
6. **Robust Scaling:** Normalized the feature space using `RobustScaler` (median and Interquartile Range, IQR) to prevent extreme outliers from distorting our model normalization parameters.

---

## Model Benchmarking Summary
We evaluated five diverse modeling families across three statistical dimensions (MAE, RMSE, and $R^2$) and measured their computational training duration on dual Tesla T4 GPUs.

### Benchmark Database
| Model Architecture | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | Goodness-of-Fit ($R^2$) | Training Duration (s) |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | 115.0110 kW | 150.6655 kW | 0.9040 | 0.9184 s |
| **Random Forest** | 19.6226 kW | 58.3702 kW | 0.9856 | 1025.0910 s |
| **XGBoost Regressor** | **16.0753 kW** | **40.6686 kW** | **0.9930** | **2.1910 s** |
| **Neural Network (DNN)** | 20.3814 kW | 48.5342 kW | 0.9900 | 35.8756 s |
| **Recurrent LSTM** | 70.5568 kW | 129.3353 kW | 0.9293 | 85.4624 s |

---

## Key Insights & Executive Recommendation

### 1. The Production Winner: GPU-Accelerated XGBoost
* **The Recommendation:** XGBoost is the definitive choice for active farm deployment. It achieved the absolute lowest prediction error (**MAE: 16.08 kW**), the lowest risk of extreme errors (**RMSE: 40.67 kW**), and explained **$99.30\%$** of the operational variance.
* **The Computational Victory:** By leveraging native CUDA acceleration, XGBoost completed training in just **$2.19$ seconds**. Compared to the CPU-bound Random Forest (which required $1,025.09$ seconds), **XGBoost trained 468 times faster while delivering superior accuracy**. In large-scale farm operations, this represents substantial savings in cloud compute spend and a minimal carbon footprint.

### 2. The Recurrent Modeling Paradox (Why the LSTM Underperformed)
* **The Takeaway:** Despite implementing a true sequence generator grouped strictly by turbine to prevent cross-contamination, the recurrent LSTM underperformed with an **MAE of 70.56 kW**.
* **The Rationale:** Wind turbine power generation is primarily an **instantaneous physical process**. The active power produced at time $T$ is mathematically determined by the physical states at that exact moment (current rotor speed, current pitch angle, current air density). Forcing a recurrent model to learn complex temporal sequences across 50 features over a 1-hour lookback window introduces optimization complexity and phase-delay noise, making simpler spatial non-linear architectures (XGBoost) vastly more effective.

---

## Quickstart Guide

### 1. Clone & Set Up Environment
```bash
# Clone the repository
git clone https://github.com/ALZ-11/Engie-turbine-analysis
cd Engie-turbine-analysis

# Create and activate a local virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install exact dependencies
pip install -r requirements.txt
```

### 2. Data Placement
Place your raw ENGIE SCADA files (`engie_X.csv` and `engie_Y.csv`) inside the `data/` folder. The internal path-resolver will automatically detect and load them:
```text
Engie-turbine-analysis/
└── data/
    ├── engie_X.csv
    └── engie_Y.csv
```

### 3. Execution
Launch Jupyter and run the end-to-end benchmarking study:
```bash
jupyter notebook notebooks/wind-turbine-analysis.ipynb
```
