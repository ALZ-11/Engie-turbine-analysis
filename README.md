# Wind Turbine Active Power Prediction: An MLOps Pipeline on the ENGIE Dataset

An end-to-end machine learning system to estimate wind turbine active power production (`TARGET`) in kilowatts (kW) using secondary physical SCADA sensor streams, intentionally excluding direct wind speed measurements.

This repository has been refactored from an exploratory notebook into a modular, test-driven MLOps architecture. It is fully serializable, protected by an automated 13-test continuous integration (CI) pipeline, and managed via an automated Retraining CLI.

---

## 1. Context & Physical Motivation
Under standard operating conditions, wind turbine active power output is modeled directly using wind speed measurements. However, physical anemometers on turbine nacelles frequently experience mechanical failure, sensor drift, or wake-effect turbulences from neighboring turbines, leading to unreliable readings.

This system implements and benchmarks an alternative modeling paradigm: **predicting active power output using only secondary physical and mechanical indicators** (shaft speeds, components temperatures, and pitch angles).

---

## 2. System Architecture & Pipeline Design
In order to guarantee system stability, prevent training-serving skew, and block data leakage, the following pipeline was established:

1. **Data Ingestion (`src/data/ingestion.py`):** Implements a dual-path loader that identifies local CSV sources first, falling back to `kagglehub` downloads. It enforces input uniqueness and row-conservation constraints to prevent row inflation during merges.

2. **Non-Linear Feature Selection (`src/features/selection.py`):**
   * **Relevance Filtering:** Computes non-parametric Mutual Information (MI) regression scores on a randomized subset to eliminate weak predictive variables and statistical noise (e.g., dropping flat-line grid frequency variables) without assuming linearity.
   * **Hierarchical Multicollinearity Pruning:** Computes pairwise Spearman rank correlation matrices, performs agglomerative complete-linkage clustering, and applies an MI-based tie-breaker to retain only the single most informative feature from each redundant group (e.g., collapsing eight redundant shaft speeds down to `Generator_speed`).

3. **Chronological Splitting (`src/data/splits.py`):** Partitions the dataset using a value-based $80/20$ chronological split utilizing the $80^{\text{th}}$ percentile of `Date_time`. This ensures duplicate timestamps across different turbines are kept together, establishing a validation boundary with zero spatiotemporal data leakage.

4. **Outlier-Robust Preprocessing (`src/features/preprocessing.py`):** Implements a Scikit-Learn `ColumnTransformer` that automates numerical median imputation, robust scaling (median and IQR-based to mitigate SCADA sensor spikes), and one-hot encoding.

5. **SCADA Gap-Aware Sequencer (`src/features/sequences.py`):** Implements a temporal sequence generator grouped by turbine that identifies chronological gaps/outages and partitions sequences into contiguous blocks. This prevents the model from learning non-physical temporal jumps.

6. **Unified Serialization:** Encapsulates the custom `ColumnDropper`, `ColumnTransformer`, and tuned estimator into a single, fully serializable Scikit-Learn `Pipeline` object exported to `models/turbine_xgb_pipeline.joblib` using `joblib`.

---

## 3. Global Model Benchmark Database
All five modeling families and our baseline model were trained on dual Tesla T4 GPUs/multi-threaded CPUs and evaluated across five statistical dimensions in original physical units (kW):

| Model Architecture | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | Goodness-of-Fit ($R^2$) | Floor Violation Rate (%) | Training Duration (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Persistence Baseline** | 69.9227 kW | 128.8066 kW | 0.9296 | 0.00% | N/A |
| **Linear Regression** | 118.6231 kW | 159.8026 kW | 0.8917 | 17.75% | 1.55s |
| **Random Forest (10 trees)** | 15.3102 kW | 39.0530 kW | 0.9935 | 0.00% | 104.93s |
| **Tuned XGBoost (Production)** | **14.8892 kW** | **35.3154 kW** | **0.9947** | **0.00%** | **4.88s** |
| **DNN** | 58.3792 kW | 96.7114 kW | 0.9603 | 0.03% | 80.21s |
| **LSTM** | 71.8580 kW | 129.3729 kW | 0.9291 | 0.01% | 337.42s |

---

## 4. Engineering & Physical Insights

### 1. Tuned XGBoost Win
* **Accuracy:** XGBoost achieved the lowest prediction error (**MAE: 14.89 kW**), the lowest risk of extreme errors (**RMSE: 35.32 kW**), and explained **$99.47\%$** of the operational variance in the testing timeframe.
* **Computational Efficiency:** By utilizing sequential gradient boosting, XGBoost completed training in just **$4.88$ seconds** on CPU. This is over $20$ times faster than building a shallow $10$-tree Random Forest, representing substantial computational cost savings during large-scale farm operations.

### 2. Recurrent Modeling Underperformance
* Despite implementing a gap-aware contiguous block sequence generator to prevent cross-contamination, the recurrent LSTM underperformed, hovering near the persistence baseline ($71.86\text{ kW}$ MAE vs. $69.92\text{ kW}$ MAE).
* Wind turbine active power generation is primarily an **instantaneous physical process**. The active power produced at time $T$ is mathematically determined by the physical forces acting on the turbine at that exact moment (current rotor speed, current pitch angle, current air density). Forcing a recurrent model to learn complex temporal sequences across 36 features over a 1-hour lookback window introduces optimization complexity and phase-delay noise, making simpler spatial non-linear architectures (XGBoost) or tuned DNNs vastly more effective.

---

## 5. Quickstart & Retraining CLI Guide

### 1. Set Up Environment
```bash
# Clone the repository
git clone https://github.com/ALZ-11/Engie-turbine-analysis
cd Engie-turbine-analysis

# Create and activate a local virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Placement
Place your raw ENGIE SCADA files (`engie_X.csv` and `engie_Y.csv`) inside the `data/` folder:
```text
Engie-turbine-analysis/
└── data/
    ├── engie_X.csv
    └── engie_Y.csv
```

### 3. Execution via Retraining CLI (`main.py`)
The command-line interface provides complete control over retraining and optimization workflows:

* **Display Help Guide:**
  ```bash
  python main.py --help
  ```

* **Optimize Hyperparameters:** Executes a 3-fold chronological walk-forward cross-validation search, logs progress, and serializes the optimal parameters to `models/best_xgb_params.yaml`:
  ```bash
  python main.py --tune
  ```

* **Train and Serialize Production Pipeline:** Ingests raw data, executes feature selection, splits chronologically, builds the unified pipeline (loading tuned parameters if present), fits the pipeline on raw features, and serializes the complete artifact to `models/turbine_xgb_pipeline.joblib`:
  ```bash
  python main.py --train
  ```

---

## 6. Automated Unit Testing & CI
The codebase is protected by a mock-driven **13-test automated suite** that tests data ingestion boundaries, information-theoretic selections, chronological splitting, target scaling, and modeling handshakes.

* **Execute Local Tests:**
  ```bash
  pytest tests/
  ```

On every push or pull request to the `main` or `master` branches, a **GitHub Actions CI runner (`.github/workflows/test_pipeline.yml`)** automatically spins up an isolated `Ubuntu` workspace, installs dependencies, and runs the entire test suite to guarantee that no regressions are introduced.