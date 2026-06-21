import joblib
from pathlib import Path
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report
from src.data_preprocessing import load_data, preprocess, split_data


MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def train_models():
    df = load_data()
    X, y_reg, y_clf, encoders, scaler = preprocess(df)
    result = split_data(X, y_reg, y_clf)
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = result

    # ── Regression models ──
    reg_models = {
        "linear_regression": LinearRegression(),
        "random_forest_regressor": RandomForestRegressor(n_estimators=100, random_state=42),
    }

    reg_results = {}
    for name, model in reg_models.items():
        model.fit(X_train, y_reg_train)
        preds = model.predict(X_test)
        reg_results[name] = {
            "model": model,
            "mae": mean_absolute_error(y_reg_test, preds),
            "r2": r2_score(y_reg_test, preds),
        }
        print(f"[{name}]  MAE: {reg_results[name]['mae']:.2f}  R²: {reg_results[name]['r2']:.3f}")

    best_reg_name = max(reg_results, key=lambda n: reg_results[n]["r2"])
    best_reg = reg_results[best_reg_name]["model"]
    print(f"\nBest regressor: {best_reg_name}")

    # ── Classification models ──
    clf_models = {
        "logistic_regression": LogisticRegression(max_iter=500, random_state=42),
        "random_forest_classifier": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    clf_results = {}
    for name, model in clf_models.items():
        model.fit(X_train, y_clf_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_clf_test, preds)
        clf_results[name] = {
            "model": model,
            "accuracy": acc,
        }
        print(f"[{name}]  Accuracy: {acc:.3f}")
        print(classification_report(y_clf_test, preds))

    best_clf_name = max(clf_results, key=lambda n: clf_results[n]["accuracy"])
    best_clf = clf_results[best_clf_name]["model"]
    print(f"Best classifier: {best_clf_name}")

    # ── Save artifacts ──
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_reg, MODELS_DIR / "regression_model.pkl")
    joblib.dump(best_clf, MODELS_DIR / "classification_model.pkl")
    joblib.dump(encoders, MODELS_DIR / "encoders.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    print("\nModels and preprocessors saved to", MODELS_DIR)

    return reg_results, clf_results


if __name__ == "__main__":
    train_models()
