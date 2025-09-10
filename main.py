import argparse, os, json, datetime
from typing import Dict, Any
from data import load_mnist, load_fashion_mnist
from models import make_pipeline
from train import ensure_dir, run_cv_and_confmat, run_grid_search

def parse_args():
    p = argparse.ArgumentParser(description="SGDClassifier on MNIST / Fashion‑MNIST with CV, augmentation, confusion matrix")
    p.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion"], help="Target dataset")
    p.add_argument("--cv", type=int, default=3, help="Number of CV folds (default: 3)")
    p.add_argument("--augment", type=int, default=0, help="Enable shift augmentation (1=yes, 0=no)")
    p.add_argument("--aug-dirs", type=int, default=4, choices=[4,8], help="Use 4 or 8 directions for shifts (default: 4)")
    p.add_argument("--shift-pixels", type=int, default=1, help="Shift in pixels (default: 1)")
    p.add_argument("--scaler", type=str, default="standard", choices=["standard", "minmax", "none"], help="Feature scaling (default: standard)")

    # SGD hyperparameters (can be overridden by --grid 1)
    p.add_argument("--loss", type=str, default="log_loss", choices=["log_loss", "hinge", "modified_huber"])
    p.add_argument("--penalty", type=str, default="l2", choices=["l2", "l1", "elasticnet"])
    p.add_argument("--alpha", type=float, default=1e-4)
    p.add_argument("--max-iter", type=int, default=2000)
    p.add_argument("--early-stopping", type=int, default=1, help="Early stopping (1=yes, 0=no)")
    p.add_argument("--n-iter-no-change", type=int, default=10)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--learning-rate", type=str, default="optimal", choices=["constant", "optimal", "invscaling", "adaptive"])
    p.add_argument("--eta0", type=float, default=0.01)

    # Grid search
    p.add_argument("--grid", type=int, default=0, help="Run small GridSearchCV first (1=yes)")

    args = p.parse_args()
    return args

def load_data(name: str):
    if name == "mnist":
        X, y = load_mnist()
        labels = [str(i) for i in range(10)]
    elif name == "fashion":
        X, y = load_fashion_mnist()
        labels = [str(i) for i in range(10)]
    else:
        raise ValueError("Unknown dataset")
    return X, y, labels

def main():
    args = parse_args()

    # Output directory
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join("runs", f"{ts}_{args.dataset}")
    ensure_dir(out_dir)

    # Save args
    with open(os.path.join(out_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    # Data
    print(f"[INFO] Loading dataset: {args.dataset}")
    X, y, labels = load_data(args.dataset)
    print(f"[INFO] X shape: {X.shape}, y shape: {y.shape}")

    # Pipeline
    sgd_params = dict(
        loss=args.loss,
        penalty=args.penalty,
        alpha=args.alpha,
        max_iter=args.max_iter,
        early_stopping=bool(args.early_stopping),
        n_iter_no_change=args.n_iter_no_change,
        tol=args.tol,
        learning_rate=args.learning_rate,
        eta0=args.eta0,
        random_state=42,
    )

    pipe = make_pipeline(
        use_augment=bool(args.augment),
        aug_dirs=args.aug_dirs,
        shift_pixels=args.shift_pixels,
        scaler=args.scaler,
        sgd_params=sgd_params,
    )

    # Optional grid search to tune SGD params (includes augmentation/scaler in the pipeline)
    best_info = None
    if args.grid:
        print("[INFO] Running grid search...")
        best_info, best_estimator = run_grid_search(pipe, X, y, cv_splits=args.cv)
        pipe = best_estimator
        with open(os.path.join(out_dir, "best_grid.json"), "w", encoding="utf-8") as f:
            json.dump(best_info, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Grid best score: {best_info['best_score']:.4f}")
        print(f"[INFO] Grid best params: {best_info['best_params']}")

    # CV + Confusion Matrix
    print(f"[INFO] Running cross‑validation (cv={args.cv}) and computing confusion matrices...")
    metrics = run_cv_and_confmat(pipe, X, y, cv_splits=args.cv, out_dir=out_dir, labels=labels)

    # Save metrics
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n=== Results ===")
    if best_info:
        print(f"GridSearchCV best acc (mean CV): {best_info['best_score']:.4f}")
    print(f"CV accuracy: {metrics['mean_accuracy']:.4f} ± {metrics['std_accuracy']:.4f}")
    print(f"Outputs saved to: {out_dir}")

if __name__ == '__main__':
    main()