# predict.py — robust to sparse/dense OHE, uses saved scalers, auto-detects K
import json
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, List

FEATURES = [
    "riceEmissionCH4", "riceEmissionN2O",
    "livestockEmissionCH4", "livestockEmissionN2O",
    "carbonSequestrationBelowGround", "carbonSequestrationAboveGround",
    "CH4Emission", "N2OEmission",
]

MODELS_DIR = Path("models")

def _dense(x) -> np.ndarray:
    """Return a dense ndarray regardless of whether x is sparse or dense."""
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)

def _load_registry() -> Dict[str, dict]:
    reg_path = MODELS_DIR / "registry.json"
    with open(reg_path, "r", encoding="utf-8") as f:
        return json.load(f)

def predict_all(ward_id: str | int,
                recent_windows: Dict[str, List[float]]) -> Dict[str, float]:
    """
    Predict next values for ALL features independently.

    recent_windows:
      { feature_name: [v_{t-k}, ..., v_{t-1}] }  # len can be >= expected k (we’ll use last k)
    """
    ward_id = str(ward_id)

    # Load shared encoder (fit during training)
    encoder = joblib.load(MODELS_DIR / "encoder.pkl")

    # One-hot encode ward id (works even if unseen because handle_unknown='ignore')
    oh = _dense(encoder.transform([[ward_id]])).ravel()
    oh_len = len(oh)

    registry = _load_registry()

    preds = {}
    for feat in FEATURES:
        info = registry.get(feat)
        if not info:
            # was skipped during training due to insufficient data
            continue

        if feat not in recent_windows:
            raise ValueError(f"Missing recent window for feature '{feat}'")

        # Load model + scalers saved by main.py
        model = joblib.load(info["model"])
        x_scaler = joblib.load(info["scaler"])
        y_scaler = joblib.load(info["y_scaler"]) if info.get("y_scaler") else None

        # Figure out how many lags (k) the model expects:
        # n_features_total = k + onehot_length
        n_features_total = x_scaler.mean_.shape[0]
        k = int(n_features_total - oh_len)
        if k <= 0:
            raise RuntimeError(f"Invalid feature layout for '{feat}': total={n_features_total}, oh={oh_len}")

        window = list(map(float, recent_windows[feat]))
        if len(window) < k:
            raise ValueError(f"Feature '{feat}' needs {k} recent values; got {len(window)}")
        if len(window) > k:
            # Accept longer input; use the most recent k values
            window = window[-k:]

        x = np.concatenate([np.asarray(window, float), oh], axis=0).reshape(1, -1)

        # Apply the same scaler used in training
        x_scaled = x_scaler.transform(x)

        # Predict and inverse-transform if y was scaled
        pred_scaled = float(model.predict(x_scaled)[0])
        if y_scaler is not None:
            pred = float(y_scaler.inverse_transform(np.array([[pred_scaled]])).ravel()[0])
        else:
            pred = pred_scaled

        preds[feat] = pred

    return preds

if __name__ == "__main__":
    # Example: main.py trained with sequence_length=4 (see main.py) → provide 4 values per feature
    example_windows = {
        "riceEmissionCH4": [100, 98, 102, 101],
        "riceEmissionN2O": [15, 16, 15.5, 15.7],
        "livestockEmissionCH4": [20, 21, 19.8, 20.2],
        "livestockEmissionN2O": [3.8, 3.7, 3.9, 3.85],
        "carbonSequestrationBelowGround": [6, 6, 6, 6],
        "carbonSequestrationAboveGround": [26, 26, 26, 26],
        "CH4Emission": [130, 131, 129, 130.5],
        "N2OEmission": [22, 21.5, 21.8, 21.6],
    }
    out = predict_all(ward_id=500, recent_windows=example_windows)
    print(out)
