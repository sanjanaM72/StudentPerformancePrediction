# Student Performance Prediction

Predict student math scores and letter grades from demographics and test scores using regression and classification models.

## Project Structure

```
StudentPerformancePrediction/
├── data/                    # Dataset (CSV)
├── src/                     # Source modules
│   ├── generate_data.py     # Synthetic data generation
│   ├── data_preprocessing.py# Encoding, scaling, train/test split
│   ├── model_training.py    # Train & save regression + classification models
│   ├── evaluation.py        # Residual, confusion matrix, feature importance plots
│   └── predict.py           # Inference API
├── notebooks/               # EDA and modeling walkthrough
├── models/                  # Saved .pkl artifacts
├── plots/                   # Evaluation charts
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# 1. Generate synthetic data
python -m src.generate_data

# 2. Train models
python -m src.model_training

# 3. Run a sample prediction
python -m src.predict
```

Or open the notebook:

```bash
jupyter notebook notebooks/01_eda_and_modeling.ipynb
```

## Models

| Task             | Algorithms                              |
|------------------|-----------------------------------------|
| Regression       | Linear Regression, Random Forest        |
| Classification   | Logistic Regression, Random Forest      |

## Results

- Regression: predicts math score (0–100) — evaluated with MAE and R²
- Classification: predicts grade (A/B/C/D/F) — evaluated with accuracy and confusion matrix
