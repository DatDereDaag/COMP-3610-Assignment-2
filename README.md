# COMP 3610 — Assignment 2: ML Model Training & Evaluation

## Overview

This project builds, evaluates, and interprets machine learning models to predict taxi trip tip amounts using the **NYC Yellow Taxi Trip dataset**. It covers feature engineering, model training and tuning with Scikit-learn, a PyTorch neural network, and comprehensive model evaluation and interpretation.

---

## Prediction Tasks

- **Regression:** Predict the continuous `tip_amount` for a given taxi trip
- **Classification:** Predict whether a trip will receive a high tip (`tip_amount > 20%` of `fare_amount`)

> Only credit card transactions (`payment_type = 1`) are used, as tip data is only reliably recorded for those payments.

---

## Project Structure

```
assignment2/
├── assignment2.ipynb       # Main notebook (Parts 1–3)
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .gitignore              # Excludes data files and model artifacts
└── data/                   # (not committed) Place dataset files here
    ├── cleaned-taxi-data.parquet
    └── taxi_zone_lookup.csv
```

---

## Setup Instructions

### Prerequisites

- Python 3.10+
- pip

### Installation

1. Clone the repository:

   ```bash
   git clone <your-repo-url>
   cd assignment2
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Place your dataset files in a `data/` folder in the project root:
   - `data/cleaned-taxi-data.parquet` — cleaned NYC Yellow Taxi data from Assignment 1
   - `data/taxi_zone_lookup.csv` — taxi zone lookup table

### Running the Notebook

Run all cells from top to bottom. Trained models will be saved to a `models/` directory (automatically created).

---

## Notebook Structure

### Part 1 — Data Preprocessing & Feature Engineering

- **Temporal features:** `pickup_hour`, `pickup_day_of_week`, `is_weekend`
- **Trip features:** `trip_duration_minutes`, `trip_speed_mph`, `log_trip_distance`
- **Fare features:** `fare_per_mile`, `fare_per_minute`
- **Zone features:** One-hot encoded pickup and dropoff borough
- Target variable creation: `tip_amount` (regression) and `high_tip` (classification)
- 70/15/15 stratified train/validation/test split with `StandardScaler`

### Part 2 — Model Training & Tuning

- **Baseline models:** Linear Regression, Random Forest Regressor, Logistic Regression, Random Forest Classifier
- **Hyperparameter tuning:** `RandomizedSearchCV` (n_iter=20, 5-fold CV) on a 200,000-row stratified sample
- **Neural network:** Feedforward network (2 hidden layers: 128 → 64) built in PyTorch for the regression task, trained for 20 epochs with `MSELoss` and Adam optimiser

### Part 3 — Model Evaluation & Interpretation

- Summary tables comparing all models on the held-out test set
- ROC curves for all classification models
- Confusion matrix for the best classifier
- Predicted vs. actual scatter plot and residual analysis for the best regressor
- Feature importance (Random Forest) and coefficient plots (Linear/Logistic Regression)
- SHAP waterfall plots for 3 sample predictions (bonus)
- Written analysis covering model performance, predictive features, limitations, and improvements

---

## Dependencies

See `requirements.txt` for full details. Key packages:

| Package      | Version  |
| ------------ | -------- |
| pandas       | >=3.0.1  |
| polars       | >=1.19.0 |
| numpy        | >=2.4.2  |
| scikit-learn | >=1.8.0  |
| torch        | >=2.5.1  |
| matplotlib   | >=3.10.8 |
| shap         | >=0.46.0 |
| joblib       | >=1.5.3  |

> **Note:** The default `torch` install is CPU-only. For GPU support, replace with the appropriate CUDA variant (e.g. `torch>=2.5.1+cu121`).
