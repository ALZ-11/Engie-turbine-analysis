import yaml
from pathlib import Path
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

from src.utils.config import PipelineConfig
from src.utils.logger import setup_logger
from src.models.classical import detect_xgb_device

logger = setup_logger()

def tune_xgboost_hyperparameters(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    config: PipelineConfig,
    n_iter: int = 10
) -> dict:
    """
    executes a time-series-aware randomized search over a defined regularization 
    and complexity space in XGBoost, saves the optimal parameters to disk.
    """
    device = detect_xgb_device()
    
    # initialize baseline estimator
    xgb_base = xgb.XGBRegressor(device=device, random_state=42, n_jobs=-1)
    
    # define time-series split cross-validation generator (walk-forward)
    logger.info("initializing 3-fold TimeSeriesSplit cross-validator...")
    tscv = TimeSeriesSplit(n_splits=3)
    
    # define the parameter search space (L1/L2 regularization and tree complexity)
    param_grid = {
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "n_estimators": [50, 100, 150],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_alpha": [0.0, 0.1, 1.0],  # L1 regularization
        "reg_lambda": [1.0, 10.0, 50.0] # L2 regularization
    }
    
    logger.info(f"initializing RandomizedSearchCV (iterations={n_iter})...")
    search_engine = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=tscv,
        scoring="neg_mean_absolute_error", # optimize for MAE
        n_jobs=-1, # parallelize cross-validation across all CPU cores
        random_state=42,
        verbose=1
    )
    
    logger.info("fitting hyperparameter search engine...")
    search_engine.fit(X_train, y_train)
    
    # extract optimal configurations
    best_params = search_engine.best_params_
    best_mae = -search_engine.best_score_
    
    logger.info("=== hyperparameter optimization successful ===")
    logger.info(f"  -> best validation MAE: {best_mae:.4f} kW")
    logger.info(f"  -> best hyperparameters: {best_params}")
    
    # serialize parameters to disk
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    params_path = models_dir / "best_xgb_params.yaml"
    
    # convert types to standard python natives for YAML serialization
    native_params = {k: int(v) if isinstance(v, (np.integer, int)) else float(v) for k, v in best_params.items()}
    
    with open(params_path, "w") as f:
        yaml.safe_dump(native_params, f)
        
    logger.info(f"optimal parameters serialized successfully to: '{params_path}'")
    return native_params