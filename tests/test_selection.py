# tests/test_selection.py
import pytest
import pandas as pd
import numpy as np
from src.utils.config import load_config
from src.features.selection import run_feature_selection_pipeline

@pytest.fixture
def mock_config():
    """loads default configurations and adjusts settings for testing."""
    config = load_config()
    # override parameters to fit the mock test data scales.
    # set a higher threshold of 0.10 to account for the finite-sample bias
    # of the k-NN entropy estimator on small (N=100) datasets.
    object.__setattr__(config.feature_selection, 'mi_sample_size', 100)
    object.__setattr__(config.feature_selection, 'mi_threshold', 0.10)
    return config


def test_run_feature_selection_pipeline(mock_config):
    """
    verifies that the feature selection pipeline correctly drops low-relevance noise,
    prunes collinear columns, and protects baseline system columns.
    """
    # seed generator for deterministic execution
    np.random.seed(42)
    
    n_rows = 100
    col_1 = np.linspace(1.0, 100.0, n_rows)
    col_2 = col_1.copy()  # col_2 is perfectly collinear with col_1
    col_noise = np.random.normal(0.0, 1.0, n_rows)  # noise column
    
    # target is strongly determined by col_1
    target = col_1 * 2.5 + np.random.normal(0.0, 5.0, n_rows)
    
    df_mock = pd.DataFrame({
        "ID": np.arange(n_rows),
        "Date_time": np.arange(n_rows),
        "MAC_CODE": ["WT1"] * n_rows,
        "Col_1": col_1,
        "Col_2": col_2,
        "Col_Noise": col_noise,
        "TARGET": target
    })
    
    # execute selection pipeline
    columns_to_drop = run_feature_selection_pipeline(df_mock, mock_config)
    
    # assertions
    # 1. col_noise should be identified as noise and dropped
    assert "Col_Noise" in columns_to_drop, "Failure: pure noise column was not identified for dropping."
    
    # 2. agglomerative clustering must identify collinearity and drop exactly one of the duplicates
    assert ("Col_1" in columns_to_drop) or ("Col_2" in columns_to_drop), "Failure: collinear duplicates were not grouped."
    assert not ("Col_1" in columns_to_drop and "Col_2" in columns_to_drop), "Failure: both collinear features were dropped instead of retaining the best one."
    
    # 3. assert system boundary columns are protected
    assert "ID" not in columns_to_drop, "Failure: core ID column was flagged for removal."
    assert "Date_time" not in columns_to_drop, "Failure: core Date_time column was flagged for removal."
    assert "MAC_CODE" not in columns_to_drop, "Failure: core MAC_CODE column was flagged for removal."
    assert "TARGET" not in columns_to_drop, "Failure: core TARGET column was flagged for removal."