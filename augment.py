from typing import Tuple, Optional
import numpy as np
from scipy.ndimage import shift as ndi_shift
from sklearn.base import BaseEstimator, TransformerMixin

_DIRS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]                      # up, down, left, right
_DIRS_8 = _DIRS_4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]          # + diagonals

class ShiftAugmenter(BaseEstimator, TransformerMixin):
    """
    Leakage-safe image shift augmenter for 28x28 images flattened to 784.
    - On fit (fit_transform): returns augmented data (original + shifts).
    - On transform (evaluation): returns X unchanged (no augmentation).
    """
    def __init__(
        self,
        shift_pixels: int = 1,
        dirs: int = 4,                 # either 4 or 8
        cval: float = 0.0,
        include_original: bool = True,
        random_state: Optional[int] = None,
    ):
        assert dirs in (4, 8), "dirs must be 4 or 8"
        self.shift_pixels = int(shift_pixels)
        self.dirs = int(dirs)
        self.cval = float(cval)
        self.include_original = bool(include_original)
        self.random_state = random_state

    def fit(self, X, y=None):
        return self

    def _shift_one(self, img2d: np.ndarray, dy: int, dx: int) -> np.ndarray:
        # scipy.ndimage.shift: shift order is (rows, cols) = (dy, dx)
        return ndi_shift(img2d, shift=(dy, dx), order=0, cval=self.cval, mode="constant", prefilter=False)

    def _augment_batch(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = X.shape[0]
        X = X.reshape(n, 28, 28)
        dirs = _DIRS_4 if self.dirs == 4 else _DIRS_8

        augmented_imgs = []
        augmented_y = []

        if self.include_original:
            augmented_imgs.append(X.copy())
            augmented_y.append(y.copy())

        for (dy, dx) in dirs:
            dy *= self.shift_pixels
            dx *= self.shift_pixels
            shifted = np.stack([self._shift_one(img, dy, dx) for img in X], axis=0)
            augmented_imgs.append(shifted)
            augmented_y.append(y.copy())

        X_aug = np.concatenate(augmented_imgs, axis=0).reshape(-1, 28*28).astype(np.float32)
        y_aug = np.concatenate(augmented_y, axis=0)
        return X_aug, y_aug

    def fit_transform(self, X, y=None, **fit_params):
        if y is None:
            return X
        X_aug, y_aug = self._augment_batch(X, y)
        return X_aug

    def transform(self, X):
        return X