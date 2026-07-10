# tests/test_preprocessing_sequences.py
import pytest
import numpy as np
import pandas as pd
from src.utils.config import load_config
from src.data.splits import split_chronologically
from src.features.preprocessing import TurbineTargetScaler
from src.features.sequences import build_gap_aware_sequences

@pytest.fixture
def mock_config():
    """loads default configurations and adjusts settings for testing."""
    config = load_config()
    # override parameters to fit the mock test data scales
    object.__setattr__(config.sequence_parameters, 'lookback_steps', 3)
    object.__setattr__(config.sequence_parameters, 'step_size', 1.0)
    return config


def test_turbine_target_scaler_reversibility():
    """verifies that the target scaling operation is mathematically reversible."""
    raw_targets = np.array([-10.5, 0.0, 150.2, 1200.4, 2250.0])
    scaler = TurbineTargetScaler()
    
    scaled = scaler.fit_transform(raw_targets)
    reversed_targets = scaler.inverse_transform(scaled)
    
    # assert within floating-point tolerance
    np.testing.assert_allclose(reversed_targets, raw_targets, rtol=1e-7)


def test_chronological_splitting_leak_prevention(mock_config):
    """verifies value-based chronological splitting prevents temporal overlap."""
    time_col = mock_config.schema.time_col
    
    # non-uniform timeline with identical timestamps representing different turbines
    df_mock = pd.DataFrame({
        time_col: [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0]
    })
    
    df_train, df_test = split_chronologically(df_mock, mock_config, train_ratio=0.6)
    
    # 60% of unique times (1, 2, 3, 4, 5) is index 3 -> split boundary is time 4.0
    # train should have times < 4.0 (i.e., 1.0, 2.0, 3.0) -> 6 rows
    # test should have times >= 4.0 (i.e., 4.0, 5.0) -> 4 rows
    assert len(df_train) == 6
    assert len(df_test) == 4
    assert df_train[time_col].max() < df_test[time_col].min()


def test_gap_aware_sequence_generation(mock_config):
    """
    verifies that the sequence builder detects gaps and skips sequences 
    that cross chronological outages.
    """
    time_col = mock_config.schema.time_col
    lookback = mock_config.sequence_parameters.lookback_steps
    
    # timeline containing a chronological gap: jump from 4.0 to 10.0
    times = [1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0, 14.0]
    n_rows = len(times)
    
    X_df_mock = pd.DataFrame({
        "MAC_CODE_WT1": [1.0] * n_rows, # representing a single turbine
        "Feature_1": np.linspace(10.0, 90.0, n_rows)
    })
    y_mock = np.linspace(100.0, 900.0, n_rows)
    df_original_mock = pd.DataFrame({time_col: times})
    
    X_seq, y_seq = build_gap_aware_sequences(X_df_mock, y_mock, df_original_mock, mock_config)
    
    # expected behavior:
    # block 1 (times 1-4, len 4): can generate (4 - lookback) = 4 - 3 = 1 sequence
    # block 2 (times 10-14, len 5): can generate (5 - lookback) = 5 - 3 = 2 sequences
    # total expected sequences = 1 + 2 = 3 (naive loop would generate 9 - 3 = 6)
    assert len(X_seq) == 3
    assert len(y_seq) == 3
    assert X_seq.shape == (3, lookback, 2) # (samples, lookback_steps, features)