import os
import time
import numpy as np
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from src.utils.config import PipelineConfig
from src.utils.logger import setup_logger
from src.features.preprocessing import TurbineTargetScaler
from src.models.metrics import calculate_evaluation_metrics

# suppress verbose TensorFlow debugging logs for cleaner terminal output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logger = setup_logger()

def ensure_model_directory() -> Path:
    """ensures 'models/' dir exists for saving artifacts."""
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def train_deep_neural_network(
    X_train: np.ndarray, 
    y_train_scaled: np.ndarray, 
    X_test: np.ndarray, 
    y_test_unscaled: np.ndarray, 
    target_scaler: TurbineTargetScaler, 
    config: PipelineConfig
) -> dict:
    """trains and scores a feedforward DNN with dropout regularization."""
    params = config.model_parameters
    model_dir = ensure_model_directory()
    model_path = str(model_dir / "best_dnn_model.keras")
    
    logger.info(f"compiling DNN architecture...")
    
    # define sequential DNN architecture
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1) # linear output layer for target regression
    ])
    
    model.compile(optimizer=Adam(learning_rate=params.dl_learning_rate), loss="mae")
    
    # configure callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss", 
            patience=params.dl_early_stopping_patience, 
            restore_best_weights=True
        ),
        ModelCheckpoint(filepath=model_path, monitor="val_loss", save_best_only=True)
    ]
    
    logger.info(f"training DNN (epochs={params.dl_epochs}, batch_size={params.dl_batch_size})...")
    start_time = time.time()
    
    # fit model on scaled targets
    model.fit(
        X_train, 
        y_train_scaled, 
        epochs=params.dl_epochs, 
        batch_size=params.dl_batch_size, 
        validation_split=0.2, 
        callbacks=callbacks, 
        verbose=1
    )
    
    elapsed_time = time.time() - start_time
    
    # generate scaled predictions and inverse-transform to kW
    y_pred_scaled = model.predict(X_test).flatten()
    y_pred_unscaled = target_scaler.inverse_transform(y_pred_scaled)
    
    metrics = calculate_evaluation_metrics(y_test_unscaled, y_pred_unscaled, params.standby_min_kw)
    metrics["Training Duration (s)"] = elapsed_time
    return {"metrics": metrics, "model": model, "predictions": y_pred_unscaled}


def train_recurrent_lstm(
    X_train_seq: np.ndarray, 
    y_train_seq_scaled: np.ndarray, 
    X_test_seq: np.ndarray, 
    y_test_seq_unscaled: np.ndarray, 
    target_scaler: TurbineTargetScaler, 
    config: PipelineConfig
) -> dict:
    """trains and scores a Recurrent LSTM network."""
    params = config.model_parameters
    model_dir = ensure_model_directory()
    model_path = str(model_dir / "best_lstm_model.keras")
    
    logger.info(f"compiling Recurrent LSTM network architecture...")
    
    # define sequential LSTM architecture
    model = Sequential([
        Input(shape=(X_train_seq.shape[1], X_train_seq.shape[2])), # (lookback, features)
        LSTM(64),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=params.dl_learning_rate), loss="mae")
    
    callbacks = [
        EarlyStopping(
            monitor="val_loss", 
            patience=params.dl_early_stopping_patience, 
            restore_best_weights=True
        ),
        ModelCheckpoint(filepath=model_path, monitor="val_loss", save_best_only=True)
    ]
    
    logger.info(f"training LSTM (epochs={params.dl_epochs}, batch_size={params.dl_batch_size})...")
    start_time = time.time()
    
    # fit model on scaled targets
    model.fit(
        X_train_seq, 
        y_train_seq_scaled, 
        epochs=params.dl_epochs, 
        batch_size=params.dl_batch_size, 
        validation_split=0.2, 
        callbacks=callbacks, 
        verbose=1
    )
    
    elapsed_time = time.time() - start_time
    
    # generate scaled predictions and inverse-transform to kW
    y_pred_scaled = model.predict(X_test_seq).flatten()
    y_pred_unscaled = target_scaler.inverse_transform(y_pred_scaled)
    
    metrics = calculate_evaluation_metrics(y_test_seq_unscaled, y_pred_unscaled, params.standby_min_kw)
    metrics["Training Duration (s)"] = elapsed_time
    return {"metrics": metrics, "model": model, "predictions": y_pred_unscaled}