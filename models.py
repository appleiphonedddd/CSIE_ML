from typing import Optional
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import SGDClassifier
from augment import ShiftAugmenter

def make_pipeline(
    use_augment: bool = False,
    aug_dirs: int = 4,
    shift_pixels: int = 1,
    scaler: str = "standard",        # 'standard' | 'minmax' | 'none'
    sgd_params: Optional[dict] = None
) -> Pipeline:
    steps = []
    if use_augment:
        steps.append(("augment", ShiftAugmenter(shift_pixels=shift_pixels, dirs=aug_dirs)))
    if scaler == "standard":
        steps.append(("scaler", StandardScaler()))
    elif scaler == "minmax":
        steps.append(("scaler", MinMaxScaler()))
    elif scaler == "none":
        pass
    else:
        raise ValueError("Unknown scaler: %s" % scaler)

    if sgd_params is None:
        sgd_params = {}

    clf = SGDClassifier(**sgd_params)
    steps.append(("sgd", clf))
    return Pipeline(steps)