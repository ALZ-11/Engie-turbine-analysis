# tests/test_ingestion.py
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.utils.config import load_config
from src.data.ingestion import load_and_merge_raw_data

@pytest.fixture
def mock_config():
    """returns standard config object for testing."""
    return load_config()

@patch("src.data.ingestion.ensure_data_sources")
@patch("pandas.read_csv")
def test_load_and_merge_raw_data_success(mock_read_csv, mock_ensure, mock_config):
    """verifies successful ingestion when data satisfies all constraints."""
    # create valid mock features and targets
    df_x_mock = pd.DataFrame({"ID": [1, 2, 3], "Feature_1": [10.0, 20.0, 30.0]})
    df_y_mock = pd.DataFrame({"ID": [1, 2, 3], "TARGET": [100.0, 200.0, 300.0]})
    
    # configure mock_read_csv to return features first, then targets
    mock_read_csv.side_effect = [df_x_mock, df_y_mock]
    
    df_merged = load_and_merge_raw_data(mock_config)
    
    assert len(df_merged) == 3
    assert "Feature_1" in df_merged.columns
    assert "TARGET" in df_merged.columns
    # ensure expected merge columns are preserved
    assert list(df_merged["ID"]) == [1, 2, 3]


@patch("src.data.ingestion.ensure_data_sources")
@patch("pandas.read_csv")
def test_load_and_merge_raw_data_duplicate_features(mock_read_csv, mock_ensure, mock_config):
    """verifies that duplicate keys in raw features halt the pipeline."""
    # ID '1' is duplicated
    df_x_mock = pd.DataFrame({"ID": [1, 1, 3], "Feature_1": [10.0, 20.0, 30.0]})
    df_y_mock = pd.DataFrame({"ID": [1, 2, 3], "TARGET": [100.0, 200.0, 300.0]})
    
    mock_read_csv.side_effect = [df_x_mock, df_y_mock]
    
    with pytest.raises(ValueError, match="raw feature IDs contains duplicate keys"):
        load_and_merge_raw_data(mock_config)


@patch("src.data.ingestion.ensure_data_sources")
@patch("pandas.read_csv")
def test_load_and_merge_raw_data_duplicate_targets(mock_read_csv, mock_ensure, mock_config):
    """verifies that duplicate keys in raw targets halt the pipeline."""
    # ID '3' is duplicated
    df_x_mock = pd.DataFrame({"ID": [1, 2, 3], "Feature_1": [10.0, 20.0, 30.0]})
    df_y_mock = pd.DataFrame({"ID": [1, 2, 3, 3], "TARGET": [100.0, 200.0, 300.0, 400.0]})
    
    mock_read_csv.side_effect = [df_x_mock, df_y_mock]
    
    with pytest.raises(ValueError, match="raw target IDs contains duplicate keys"):
        load_and_merge_raw_data(mock_config)


@patch("src.data.ingestion.ensure_data_sources")
@patch("pandas.read_csv")
def test_load_and_merge_raw_data_cardinality_mismatch(mock_read_csv, mock_ensure, mock_config):
    """verifies that mismatched IDs (orphan keys) trigger row-conservation failure."""
    # ID '4' exists in features but is missing from targets, which will drop rows in an inner join
    df_x_mock = pd.DataFrame({"ID": [1, 2, 4], "Feature_1": [10.0, 20.0, 30.0]})
    df_y_mock = pd.DataFrame({"ID": [1, 2, 3], "TARGET": [100.0, 200.0, 300.0]})
    
    mock_read_csv.side_effect = [df_x_mock, df_y_mock]
    
    with pytest.raises(AssertionError, match="inner join operations altered the dataset cardinality"):
        load_and_merge_raw_data(mock_config)