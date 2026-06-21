import joblib
import pandas as pd
import numpy as np
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class StudentPerformancePredictor:
    def __init__(self):
        self.reg_model = joblib.load(MODELS_DIR / "regression_model.pkl")
        self.clf_model = joblib.load(MODELS_DIR / "classification_model.pkl")
        self.encoders = joblib.load(MODELS_DIR / "encoders.pkl")
        self.scaler = joblib.load(MODELS_DIR / "scaler.pkl")

    def predict(self, student_data: dict) -> dict:
        df = pd.DataFrame([student_data])
        for col, le in self.encoders.items():
            if col in df.columns:
                val = df[col].iloc[0]
                if val not in le.classes_:
                    le.classes_ = np.append(le.classes_, val)
                df[col] = le.transform(df[col])
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = self.scaler.transform(df[num_cols])

        math_pred = self.reg_model.predict(df)[0]
        math_pred = int(round(np.clip(math_pred, 0, 100)))
        grade_pred = self.clf_model.predict(df)[0]

        return {"predicted_math_score": math_pred, "predicted_grade": grade_pred}


if __name__ == "__main__":
    sample = {
        "gender": "female",
        "race_ethnicity": "group B",
        "parental_education": "bachelor's",
        "lunch": "standard",
        "test_prep": "completed",
        "reading_score": 85,
        "writing_score": 82,
    }
    predictor = StudentPerformancePredictor()
    result = predictor.predict(sample)
    print("Sample prediction:", result)
