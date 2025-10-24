# predict.py — batch forecasting from a data file, one sheet per future year
import argparse
import json
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import deque, defaultdict
import pandas as pd

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

def _load_models_and_k(encoder) -> Tuple[Dict[str, object], Dict[str, object], Dict[str, object], Dict[str, int]]:
    """
    Load per-feature model, x_scaler, (optional) y_scaler, and infer k (lags) from scaler dimensionality.
    Returns dicts keyed by feature, and a dict feature->k.
    """
    registry = _load_registry()
    models, xscalers, yscalers, ks = {}, {}, {}, {}
    oh_len = len(encoder.categories_[0])

    for feat in FEATURES:
        info = registry.get(feat)
        if not info:
            continue  # skipped at training due to insufficient data

        model = joblib.load(info["model"])
        x_scaler = joblib.load(info["scaler"])
        y_scaler = joblib.load(info["y_scaler"]) if info.get("y_scaler") else None

        # Infer k: n_total_features = k + onehot_len
        n_total = x_scaler.mean_.shape[0]
        k = int(n_total - oh_len)
        if k <= 0:
            # Invalid layout for this feature; skip it
            continue

        models[feat] = model
        xscalers[feat] = x_scaler
        yscalers[feat] = y_scaler
        ks[feat] = k

    return models, xscalers, yscalers, ks

def predict_all(ward_id: str | int,
                recent_windows: Dict[str, List[float]]) -> Dict[str, float]:
    """
    (Kept for single-ward usage)
    Predict next values for ALL features independently for a single ward.

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

# ============ NEW: batch forecasting from a file ============
def _build_initial_windows(df: pd.DataFrame,
                           usable_features: List[str],
                           k_by_feat: Dict[str, int]) -> Tuple[dict, dict, pd.DataFrame, pd.Series]:
    """
    Prepare rolling windows per (ward, feature) of length k.
    Returns:
      rolling[wid][feat] = deque(length=k)
      series[wid][feat]  = original series list (by year)
      ward_means, global_means for padding
    """
    df_sorted = df.sort_values(["id", "year"]).copy()

    # Means for padding
    ward_means = df_sorted.groupby("id")[usable_features].mean(numeric_only=True)
    global_means = df_sorted[usable_features].mean(numeric_only=True)

    # Organize time series by ward
    series = defaultdict(lambda: defaultdict(list))
    for wid, grp in df_sorted.groupby("id"):
        grp = grp.sort_values("year")
        for feat in usable_features:
            series[wid][feat] = grp[feat].astype(float).tolist()

    # Initialize rolling windows
    rolling = defaultdict(dict)
    for wid in series.keys():
        for feat in usable_features:
            k = k_by_feat[feat]
            vals = series[wid][feat]
            if len(vals) >= k:
                base = vals[-k:]
            else:
                base = vals[:]
                need = k - len(base)
                wm = ward_means.loc[wid, feat] if wid in ward_means.index else np.nan
                gm = global_means[feat]
                pad_val = wm if not pd.isna(wm) else gm
                base = ([float(pad_val)] * need) + [float(x) for x in base]
            rolling[wid][feat] = deque(base, maxlen=k)

    return rolling, series, ward_means, global_means

def forecast_from_file(data_path: str | Path,
                       out_path: str | Path,
                       n_years: int = 5) -> Path:
    """
    Read the input Excel, forecast next n_years (global last year + 1..+n),
    and write an Excel workbook with one sheet per forecast year.
    """
    data_path = Path(data_path)
    out_path = Path(out_path)

    # Load data
    df = pd.read_excel(data_path)
    need_cols = {"id", "year"}
    if not need_cols.issubset(df.columns):
        raise ValueError(f"Input file must contain columns {sorted(need_cols)}")

    # Keep only columns we have models for (and that exist in file)
    registry = _load_registry()
    trained_features = [k for k, v in registry.items() if v is not None]
    usable_features = [c for c in trained_features if c in df.columns]
    if not usable_features:
        raise RuntimeError("No trained features found in the input file.")

    df = df[["id", "year"] + usable_features].copy()
    df["id"] = df["id"].astype(str)
    df["year"] = df["year"].astype(int)

    # Determine forecast horizon years by GLOBAL last year (shared sheets)
    global_last_year = int(df["year"].max())
    forecast_years = [global_last_year + i for i in range(1, n_years + 1)]

    # Load encoder & per-feature models/scalers; infer k per feature
    encoder = joblib.load(MODELS_DIR / "encoder.pkl")
    models, xscalers, yscalers, k_by_feat = _load_models_and_k(encoder)

    # Remove any feature that failed to load or has bad k
    usable_features = [f for f in usable_features if f in models and f in xscalers and f in k_by_feat]
    if not usable_features:
        raise RuntimeError("No usable trained features after loading models/scalers.")

    # Build initial rolling windows per (ward, feature)
    rolling, series, _, _ = _build_initial_windows(df, usable_features, k_by_feat)

    # Forecast loop (year by year), one sheet per year
    sheets = {}
    for y in forecast_years:
        rows = []
        for wid in series.keys():
            row = {"id": wid, "year": y}
            # Prepare OHE for this ward
            oh = _dense(encoder.transform([[str(wid)]]))
            oh = np.asarray(oh).ravel()

            for feat in usable_features:
                model = models[feat]
                x_scaler = xscalers[feat]
                y_scaler = yscalers[feat]
                window = list(rolling[wid][feat])  # length k, chronological

                x = np.concatenate([np.asarray(window, float), oh], axis=0).reshape(1, -1)
                xs = x_scaler.transform(x)

                pred_scaled = float(model.predict(xs)[0])
                if y_scaler is not None:
                    pred = float(y_scaler.inverse_transform(np.array([[pred_scaled]])).ravel()[0])
                else:
                    pred = pred_scaled

                row[feat] = pred
                # roll forward for next steps
                rolling[wid][feat].append(pred)

            rows.append(row)

        sheet_df = pd.DataFrame(rows, columns=["id", "year"] + usable_features)
        sheets[y] = sheet_df

    # Write to Excel: one sheet per year
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        for y in forecast_years:
            sheets[y].to_excel(writer, sheet_name=str(y), index=False)

    return out_path

# ---------------- CLI ----------------
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Forecast next N years for every ward and write one sheet per year.")
    p.add_argument("--data", default="../new_data/ward_emission_cleaned.xlsx", help="Path to input Excel with columns: id, year, and trained feature columns.")
    p.add_argument("--out", default="../output_predict/predict.xlsx", help="Path to output Excel (e.g., forecast_next5.xlsx).")
    p.add_argument("--years", type=int, default=5, help="Number of consecutive years to forecast (default: 5).")
    return p

if __name__ == "__main__":
    # Example single-ward usage (kept for reference)
    # recent_windows_example = {"area": [..k..], ...}
    # print(predict_all(ward_id=500, recent_windows=recent_windows_example))

    parser = _build_argparser()
    args = parser.parse_args()
    out_path = forecast_from_file(args.data, args.out, n_years=args.years)
    print(f"✅ Wrote forecasts to: {out_path}")
