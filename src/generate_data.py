import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

N = 2000

data = {
    "gender": np.random.choice(["male", "female"], N, p=[0.48, 0.52]),
    "race_ethnicity": np.random.choice(
        ["group A", "group B", "group C", "group D", "group E"], N,
        p=[0.08, 0.25, 0.35, 0.22, 0.10]
    ),
    "parental_education": np.random.choice(
        ["some high school", "high school", "some college", "associate's",
         "bachelor's", "master's", "doctorate"], N,
        p=[0.10, 0.25, 0.20, 0.15, 0.15, 0.10, 0.05]
    ),
    "lunch": np.random.choice(["standard", "free/reduced"], N, p=[0.65, 0.35]),
    "test_prep": np.random.choice(["none", "completed"], N, p=[0.60, 0.40]),
    "reading_score": np.random.normal(70, 15, N).clip(0, 100).astype(int),
    "writing_score": np.random.normal(70, 15, N).clip(0, 100).astype(int),
}

df = pd.DataFrame(data)

lunch_effect = np.where(df["lunch"] == "standard", 5, -3)
test_prep_effect = np.where(df["test_prep"] == "completed", 6, -2)
noise = np.random.normal(0, 6, N)

base = (
    0.40 * df["reading_score"]
    + 0.40 * df["writing_score"]
    + lunch_effect
    + test_prep_effect
    + noise
)
df["math_score"] = (base - 5).clip(0, 100).astype(int)

def assign_grade(score):
    if score >= 85: return "A"
    if score >= 70: return "B"
    if score >= 55: return "C"
    if score >= 40: return "D"
    return "F"

df["grade"] = df["math_score"].apply(assign_grade)

out = Path(__file__).resolve().parent.parent / "data"
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "student_performance.csv", index=False)
print(f"Created {out / 'student_performance.csv'} with {N} records")
