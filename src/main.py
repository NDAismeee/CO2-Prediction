# main.py
from pathlib import Path
import json
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from data import load_and_prepare_multi
from model import train_and_save_model

excel_path = "../new_data/ward_emission_cleaned.xlsx"
sequence_length = 4
LAST_N_YEARS_TEST = 2   # how many last years per id go to test

FEATURES = [
    "riceEmissionCH4", "riceEmissionN2O",
    "livestockEmissionCH4", "livestockEmissionN2O",
    "carbonSequestrationBelowGround", "carbonSequestrationAboveGround",
    "CH4Emission", "N2OEmission",
]

# Targets to scale (you can change this to set(FEATURES) to scale all)
SCALE_Y_FOR = {
    "carbonSequestrationBelowGround",
    "carbonSequestrationAboveGround",
}

models_dir = Path("models")
models_dir.mkdir(parents=True, exist_ok=True)

# Now returns X_map, y_map, meta_map, encoder
X_map, Y_map, META_map, encoder = load_and_prepare_multi(
    filepath=excel_path,
    feature_cols=FEATURES,
    sequence_length=sequence_length,
    id_col="id",
    year_col="year",
)

# Save encoder once (shared id space)
joblib.dump(encoder, models_dir / "encoder.pkl")
print(f"✅ Saved encoder to {models_dir/'encoder.pkl'}")

registry = {}

for feat in FEATURES:
    X, y, meta = X_map[feat], Y_map[feat], META_map[feat]
    if len(X) < 2:
        print(f"[warn] Not enough samples for '{feat}' — skipping.")
        continue

    # ---- Time-aware split: last N years per id = test ----
    test_mask = np.zeros(len(meta), dtype=bool)
    for _id, grp in meta.groupby("id"):
        years_sorted = np.sort(grp["target_year"].unique())
        last_n = years_sorted[-LAST_N_YEARS_TEST:] if len(years_sorted) >= LAST_N_YEARS_TEST else years_sorted
        mask = grp["target_year"].isin(last_n).values
        test_mask[grp.index] = mask

    X_train, X_test = X[~test_mask], X[test_mask]
    y_train, y_test = y[~test_mask], y[test_mask]

    if len(X_train) == 0 or len(X_test) == 0:
        print(f"[warn] '{feat}': empty split, fallback to last-sample split")
        X_train, X_test = X[:-1], X[-1:]
        y_train, y_test = y[:-1], y[-1:]

    # === Normalize X ===
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled  = x_scaler.transform(X_test)
    x_scaler_path = models_dir / f"scaler_{feat}.pkl"
    joblib.dump(x_scaler, x_scaler_path)

    # === (Optional) Normalize y ===
    if feat in SCALE_Y_FOR:
        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test_scaled  = y_scaler.transform(y_test.reshape(-1, 1)).ravel()
        y_scaler_path = models_dir / f"y_scaler_{feat}.pkl"
        joblib.dump(y_scaler, y_scaler_path)
        save_y_scaler_path = str(y_scaler_path)
    else:
        y_scaler = None
        y_train_scaled = y_train
        y_test_scaled  = y_test
        save_y_scaler_path = None

    # Train model on (X_train_scaled, y_train_scaled)
    model_path = models_dir / f"model_{feat}.pkl"
    model = train_and_save_model(
        X_train_scaled, y_train_scaled,
        save_path=str(model_path),
        use_ridge=True,
        alpha=1.0,
    )

    # Evaluate (inverse-transform preds if we scaled y)
    train_pred = model.predict(X_train_scaled)
    test_pred  = model.predict(X_test_scaled)

    if y_scaler is not None:
        train_pred_eval = y_scaler.inverse_transform(train_pred.reshape(-1, 1)).ravel()
        test_pred_eval  = y_scaler.inverse_transform(test_pred.reshape(-1, 1)).ravel()
        y_train_eval, y_test_eval = y_train, y_test
    else:
        train_pred_eval, test_pred_eval = train_pred, test_pred
        y_train_eval, y_test_eval = y_train, y_test

    train_mae  = mean_absolute_error(y_train_eval, train_pred_eval)
    test_mae   = mean_absolute_error(y_test_eval,  test_pred_eval)
    train_rmse = np.sqrt(mean_squared_error(y_train_eval, train_pred_eval))
    test_rmse  = np.sqrt(mean_squared_error(y_test_eval,  test_pred_eval))

    print(f"📊 [{feat}] Train MAE {train_mae:.4f} RMSE {train_rmse:.4f}")
    print(f"📈 [{feat}] Test  MAE {test_mae:.4f} RMSE {test_rmse:.4f}  (n_test={len(X_test)})")

    registry[feat] = {
        "model": str(model_path),
        "scaler": str(x_scaler_path),
        "y_scaler": save_y_scaler_path,       # None if not used
        "last_n_years_test": LAST_N_YEARS_TEST,
    }

with open(models_dir / "registry.json", "w", encoding="utf-8") as f:
    json.dump(registry, f, ensure_ascii=False, indent=2)

print("✅ Trained models + scalers with time-aware split & target scaling (where enabled). See models/registry.json")
