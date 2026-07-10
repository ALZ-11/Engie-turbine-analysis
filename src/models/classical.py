import os
import time
import yaml
from pathlib import Path
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from src.utils.config import PipelineConfig
from src.utils.logger import setup_logger
from src.models.metrics import calculate_evaluation_metrics

logger = setup_logger()

def detect_xgb_device() -> str:
    """
    attempts to train 1-row dataset to verify if CUDA GPU
    acceleration is supported and allocated on the host hardware.
    """
    try:
        X_dummy = np.array([[1.0]])
        y_dummy = np.array([1.0])
        test_model = xgb.XGBRegressor(n_estimators=1, device="cuda")
        test_model.fit(X_dummy, y_dummy)
        logger.info("GPU CUDA acceleration successfully validated. XGBoost will utilize GPU.")
        return "cuda"
    except Exception:
        logger.warning("GPU acceleration (CUDA) not available or supported. Falling back to multi-threaded CPU.")
        return "cpu"


def train_linear_regression(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, config: PipelineConfig
) -> dict:
    """trains and scores a baseline Linear Regression model."""
    logger.info("training Linear Regression...")
    model = LinearRegression()
    
    start_time = time.time()
    model.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    
    y_pred = model.predict(X_test)
    metrics = calculate_evaluation_metrics(y_test, y_pred, config.model_parameters.standby_min_kw)
    metrics["Training Duration (s)"] = elapsed_time
    return {"metrics": metrics, "model": model, "predictions": y_pred}


def train_random_forest(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, config: PipelineConfig
) -> dict:
    """trains and scores a multi-threaded CPU-bound RF regressor."""
    params = config.model_parameters
    logger.info(f"training Random Forest on CPU (n_estimators={params.rf_n_estimators}, max_depth={params.rf_max_depth})...")
    
    model = RandomForestRegressor(
        n_estimators=params.rf_n_estimators,
        max_depth=params.rf_max_depth,
        random_state=42,
        n_jobs=-1 # force multi-threaded core saturation
    )
    
    start_time = time.time()
    model.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    
    y_pred = model.predict(X_test)
    metrics = calculate_evaluation_metrics(y_test, y_pred, params.standby_min_kw)
    metrics["Training Duration (s)"] = elapsed_time
    return {"metrics": metrics, "model": model, "predictions": y_pred}


def train_xgboost(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, config: PipelineConfig
) -> dict:
    """
    trains and scores an XGBoost Regressor.
    loads optimized hyperparameters from disk if they exist;
    falls back to default configurations otherwise.
    """
    params = config.model_parameters
    device = detect_xgb_device()
    
    # define fallback parameters (default)
    model_kwargs = {
        "n_estimators": params.xgb_n_estimators,
        "learning_rate": params.xgb_learning_rate,
        "device": device,
        "random_state": 42,
        "n_jobs": -1
    }
    
    # attempt to load tuned hyperparameters from disk
    tuned_params_path = Path("models/best_xgb_params.yaml")
    if tuned_params_path.exists():
        try:
            with open(tuned_params_path, "r") as f:
                tuned_kwargs = yaml.safe_load(f)
            # inject device and safety arguments into loaded kwargs
            tuned_kwargs["device"] = device
            tuned_kwargs["random_state"] = 42
            tuned_kwargs["n_jobs"] = -1
            
            model_kwargs = tuned_kwargs
            logger.info(f"tuned hyperparameters successfully loaded from: '{tuned_params_path}'")
        except Exception as e:
            logger.warning(f"failed to parse tuned parameters, falling back to config defaults. Error: {e}")
    else:
        logger.info("no tuned parameters found on disk, proceeding with configuration defaults.")
        
    logger.info(f"training XGBoost model...")
    model = xgb.XGBRegressor(**model_kwargs)
    
    start_time = time.time()
    model.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    
    y_pred = model.predict(X_test)
    metrics = calculate_evaluation_metrics(y_test, y_pred, params.standby_min_kw)
    metrics["Training Duration (s)"] = elapsed_time
    return {"metrics": metrics, "model": model, "predictions": y_pred}