from typing import Dict, Any, Tuple
import os, json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict, GridSearchCV

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def run_cv_and_confmat(pipeline, X, y, cv_splits: int, out_dir: str, labels=None) -> Dict[str, Any]:
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    # CV accuracy
    cv_scores = cross_val_score(pipeline, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
    mean_acc = float(np.mean(cv_scores))
    std_acc = float(np.std(cv_scores))

    # Confusion matrix via OOF predictions (uses the same splits)
    y_pred = cross_val_predict(pipeline, X, y, cv=skf, n_jobs=-1)
    cm = confusion_matrix(y, y_pred, labels=labels)
    cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    # Save CM plots
    fig1 = plot_confusion_matrix(cm, labels, normalize=False, title="Confusion Matrix")
    fig1.savefig(os.path.join(out_dir, "confusion_matrix.png"), bbox_inches="tight", dpi=160)
    plt.close(fig1)

    fig2 = plot_confusion_matrix(cm_norm, labels, normalize=True, title="Confusion Matrix (Row‑Normalized)")
    fig2.savefig(os.path.join(out_dir, "confusion_matrix_normalized.png"), bbox_inches="tight", dpi=160)
    plt.close(fig2)

    # Classification report
    report = classification_report(y, y_pred, labels=labels, digits=4, zero_division=0)
    with open(os.path.join(out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    return {
        "cv_scores": list(map(float, cv_scores)),
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
    }

def plot_confusion_matrix(matrix: np.ndarray, labels=None, normalize=False, title="Confusion Matrix"):
    fig = plt.figure(figsize=(7.5, 6.0))
    ax = fig.add_subplot(111)
    im = ax.imshow(matrix, interpolation="nearest")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if labels is None:
        labels = list(range(matrix.shape[0]))
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    # Annotate for small matrices
    if matrix.shape[0] <= 10 and matrix.shape[1] <= 10:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                txt = f"{val:.2f}" if normalize else str(int(val))
                ax.text(j, i, txt, ha="center", va="center")
    fig.tight_layout()
    return fig

def run_grid_search(pipeline, X, y, cv_splits: int) -> Tuple[Dict[str, Any], Any]:
    """
    Runs a small grid search over SGD hyperparameters. Returns (best_info, best_estimator).
    """
    param_grid = {
        "sgd__loss": ["log_loss", "hinge", "modified_huber"],
        "sgd__penalty": ["l2", "elasticnet"],
        "sgd__alpha": [1e-4, 1e-3, 1e-2],
        "sgd__learning_rate": ["optimal", "adaptive"],
        "sgd__eta0": [0.1, 0.01],
        "sgd__max_iter": [2000],
        "sgd__early_stopping": [True],
        "sgd__n_iter_no_change": [10],
    }
    gs = GridSearchCV(
        pipeline, param_grid=param_grid, scoring="accuracy",
        cv=cv_splits, n_jobs=-1, verbose=1, refit=True
    )
    gs.fit(X, y)
    best = {
        "best_score": float(gs.best_score_),
        "best_params": gs.best_params_,
    }
    return best, gs.best_estimator_