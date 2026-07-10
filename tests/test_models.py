# tests/test_models.py
import pytest
import yaml
from pathlib import Path
import numpy as np
from src.utils.config import load_config
from src.features.preprocessing import TurbineTargetScaler
from src.models.classical import train_linear_regression, train_xgboost, detect_xgb_device
from src.models.deep_learning import train_deep_neural_network, train_recurrent_lstm
from src.models.tuning import tune_xgboost_hyperparameters

@pytest.fixture
def mock_model_config():
    """loads default configurations and minimizes epoch scales for instant testing."""
    config = load_config()
    # override parameters to execute instantly
    object.__setattr__(config.model_parameters, 'dl_epochs', 1)
    object.__setattr__(config.model_parameters, 'dl_batch_size', 10)
    object.__setattr__(config.model_parameters, 'xgb_n_estimators', 1)
    return config


def test_detect_xgb_device():
    """verifies that the device detection function returns a valid string."""
    device = detect_xgb_device()
    assert device in ["cuda", "cpu"]


def test_classical_model_training_handshake(mock_model_config):
    """verifies that classical training functions return correct data structures."""
    # create small mock data
    X_train = np.random.normal(0, 1, (50, 10))
    y_train = np.random.normal(100, 50, 50)
    X_test = np.random.normal(0, 1, (10, 10))
    y_test = np.random.normal(100, 50, 10)
    
    # test linear regression
    lr_results = train_linear_regression(X_train, y_train, X_test, y_test, mock_model_config)
    assert "metrics" in lr_results
    assert "model" in lr_results
    assert len(lr_results["predictions"]) == 10
    
    # test xgboost
    xgb_results = train_xgboost(X_train, y_train, X_test, y_test, mock_model_config)
    assert "metrics" in xgb_results
    assert "model" in xgb_results
    assert len(xgb_results["predictions"]) == 10


def test_deep_neural_network_training_handshake(mock_model_config):
    """verifies DNN trains, predicts, and inverse-transforms correctly."""
    X_train = np.random.normal(0, 1, (50, 36))
    y_train_raw = np.random.normal(100, 50, 50)
    X_test = np.random.normal(0, 1, (10, 36))
    y_test_raw = np.random.normal(100, 50, 10)
    
    # target scaler
    target_scaler = TurbineTargetScaler()
    y_train_scaled = target_scaler.fit_transform(y_train_raw)
    
    dnn_results = train_deep_neural_network(
        X_train, y_train_scaled, X_test, y_test_raw, target_scaler, mock_model_config
    )
    
    assert "metrics" in dnn_results
    assert "model" in dnn_results
    assert len(dnn_results["predictions"]) == 10
    # predictions must be inverse-transformed back to physical kW range (not normalized)
    assert np.max(dnn_results["predictions"]) > 5.0


def test_recurrent_lstm_training_handshake(mock_model_config):
    """verifies LSTM trains, predicts, and inverse-transforms correctly."""
    # (samples, lookback, features)
    X_train_seq = np.random.normal(0, 1, (50, 6, 36))
    y_train_seq_scaled = np.random.normal(0, 1, 50)
    X_test_seq = np.random.normal(0, 1, (10, 6, 36))
    y_test_seq_unscaled = np.random.normal(100, 50, 10)
    
    target_scaler = TurbineTargetScaler()
    # dummy fit to allow inverse transformation
    target_scaler.fit(y_test_seq_unscaled)
    
    lstm_results = train_recurrent_lstm(
        X_train_seq, y_train_seq_scaled, X_test_seq, y_test_seq_unscaled, target_scaler, mock_model_config
    )
    
    assert "metrics" in lstm_results
    assert "model" in lstm_results
    assert len(lstm_results["predictions"]) == 10


def test_xgboost_hyperparameter_tuning_and_serialization(mock_model_config):
    """
    verifies that the tuning engine runs correctly, outputs a valid YAML parameter file,
    and tears down afterward.
    """
    X_train = np.random.normal(0, 1, (30, 10))
    y_train = np.random.normal(100, 50, 30)
    
    tuned_params_path = Path("models/best_xgb_params.yaml")
    
    # backup existing parameters if they exist to prevent tests from overwriting user progress
    backup_data = None
    if tuned_params_path.exists():
        with open(tuned_params_path, "r") as f:
            backup_data = yaml.safe_load(f)
            
    try:
        # run optimization (1 iteration)
        best_params = tune_xgboost_hyperparameters(X_train, y_train, mock_model_config, n_iter=1)
        
        # verify the file was written and is syntactically valid
        assert tuned_params_path.exists()
        with open(tuned_params_path, "r") as f:
            loaded_params = yaml.safe_load(f)
            
        assert "max_depth" in loaded_params
        assert "learning_rate" in loaded_params
        assert loaded_params["max_depth"] == best_params["max_depth"]
        
    finally:
        # clean up and restore state
        if tuned_params_path.exists():
            tuned_params_path.unlink()
        if backup_data is not None:
            # restore user's tuned parameters
            with open(tuned_params_path, "w") as f:
                yaml.safe_dump(backup_data, f)