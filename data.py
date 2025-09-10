from typing import Tuple
import numpy as np
from sklearn.datasets import fetch_openml

def _as_uint8_float(X: np.ndarray) -> np.ndarray:
    # Ensure float32 in [0,1]
    X = X.astype(np.float32)
    if X.max() > 1.0:
        X /= 255.0
    return X

def load_mnist() -> Tuple[np.ndarray, np.ndarray]:
    """Load MNIST from OpenML; returns X (n, 784), y (n,) as strings '0'..'9'."""
    data = fetch_openml(name="mnist_784", version=1, as_frame=False, parser="auto")
    X, y = data.data, data.target
    X = _as_uint8_float(X)
    return X, y

def load_fashion_mnist() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load Fashion-MNIST from OpenML with robust name fallbacks.
    Returns X (n, 784), y (n,) as class labels '0'..'9'.
    """
    candidates = [
        ("Fashion-MNIST", 1),
        ("Fashion-MNIST", None),
        ("Fashion-MNIST (original)", None),
        ("fashion-mnist", None),
    ]
    last_err = None
    for name, ver in candidates:
        try:
            data = fetch_openml(name=name, version=ver, as_frame=False, parser="auto")
            X, y = data.data, data.target
            X = _as_uint8_float(X)
            return X, y
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Unable to load Fashion-MNIST via OpenML. Last error: {last_err}")