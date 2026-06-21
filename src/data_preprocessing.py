import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path


def load_data(path=None):
    if path is None:
        path = Path(__file__).resolve().parent.parent / "data" / "student_performance.csv"
    df = pd.read_csv(path)
    return df


def preprocess(df, target="math_score"):
    X = df.drop(columns=[target, "grade"], errors="ignore")
    y_reg = df[target] if target in df.columns else None
    y_clf = df.get("grade", None)

    cat_cols = X.select_dtypes(include="object").columns
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    num_cols = X.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    return X, y_reg, y_clf, encoders, scaler


def split_data(X, y_reg, y_clf=None, test_size=0.2, random_state=42):
    if y_clf is not None:
        X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
            X, y_reg, y_clf, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test
    X_train, X_test, y_reg_train, y_reg_test = train_test_split(
        X, y_reg, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_reg_train, y_reg_test


import numpy as np
