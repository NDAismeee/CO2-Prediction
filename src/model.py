import joblib
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import os


def train_and_save_model(X, y, save_path="models/buffalo_lr.pkl", use_ridge=False, alpha=1.0):
    """
    Train a Linear or Ridge Regression model and save it to disk.

    Args:
        X: ndarray, shape (samples, features)
        y: ndarray, shape (samples,)
        save_path: path to save the model
        use_ridge: bool, if True use Ridge instead of Linear Regression
        alpha: float, regularization strength for Ridge

    Returns:
        model: trained model
    """
    if use_ridge:
        model = Ridge(alpha=alpha)
        print("🔧 Using Ridge Regression")
    else:
        model = LinearRegression()
        print("🔧 Using Linear Regression")

    model.fit(X, y)

    # Evaluate
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    print(f"✅ Training done. MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    print(f"✅ Model saved to {save_path}")

    return model
