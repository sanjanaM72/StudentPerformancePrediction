import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

sns.set_style("whitegrid")


def plot_residuals(y_true, y_pred, model_name="model"):
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].scatter(y_pred, residuals, alpha=0.5)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residual")
    axes[0].set_title(f"{model_name} — Residuals")

    axes[1].hist(residuals, bins=30, edgecolor="black", alpha=0.7)
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"{model_name} — Residual Distribution")

    plt.tight_layout()
    path = PLOTS_DIR / f"{model_name}_residuals.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved {path}")


def plot_confusion(y_true, y_pred, model_name="model"):
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap="Blues")
    ax.set_title(f"{model_name} — Confusion Matrix")
    plt.tight_layout()
    path = PLOTS_DIR / f"{model_name}_confusion.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved {path}")


def plot_feature_importance(model, feature_names, model_name="model", top_n=10):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
        if importances.ndim > 1:
            importances = importances.mean(axis=0)
    else:
        print(f"No feature importance for {model_name}")
        return

    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(idx)), importances[idx], color="steelblue")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx])
    ax.set_xlabel("Importance")
    ax.set_title(f"{model_name} — Top {top_n} Features")
    plt.tight_layout()
    path = PLOTS_DIR / f"{model_name}_feature_importance.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved {path}")
