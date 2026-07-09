import os
import pandas as pd
from src.utils.config import PipelineConfig
from src.utils.logger import setup_logger

logger = setup_logger()

def ensure_data_sources(config: PipelineConfig) -> None:
    """
    ensures raw data source csvs exist. 
	if missing, attempts to download them from Kaggle using kagglehub.
    """
    x_exists = config.paths.raw_x_path.exists()
    y_exists = config.paths.raw_y_path.exists()
    
    if x_exists and y_exists:
        logger.info("local raw SCADA data sources validated.")
        return
        
    logger.warning("local raw data files not found in standard paths, attempting Kaggle fallback...")
    try:
        import kagglehub
        download_path = kagglehub.dataset_download("loziadam/engiedata")
        logger.info(f"dataset successfully downloaded via kagglehub to: '{download_path}'")
        
        # override config path to point to the downloaded directory
        # (temporary for this run)
        object.__setattr__(config.paths, 'data_dir', config.paths.data_dir.parent / download_path)
        # re-resolve paths based on downloaded files
        # handle file-finding depending on structural differences:
        for file in os.listdir(download_path):
            if "engie_X" in file:
                object.__setattr__(config.paths, 'raw_x_path', config.paths.data_dir / file)
            elif "engie_Y" in file:
                object.__setattr__(config.paths, 'raw_y_path', config.paths.data_dir / file)
                
    except Exception as e:
        logger.error(f"Kaggle API download fallback failed: {e}")
        raise FileNotFoundError(
            "raw SCADA files missing locally and remote backup download failed. "
            "ensure 'engie_X.csv' and 'engie_Y.csv' are present under the raw data path directory."
        )


def load_and_merge_raw_data(config: PipelineConfig) -> pd.DataFrame:
    """
    loads engie_X and engie_Y, verifies unique structural constraints,
    validates baseline schema compliance, and performs an inner merge on ID.
    """
    # path & file presence check
    ensure_data_sources(config)
    
    logger.info(f"loading raw features from: '{config.paths.raw_x_path}'")
    logger.info(f"loading raw targets from: '{config.paths.raw_y_path}'")
    
    # ingest raw files
    # (SCADA files utilize ';' separators)
    df_x = pd.read_csv(config.paths.raw_x_path, sep=";")
    df_y = pd.read_csv(config.paths.raw_y_path, sep=";")
    
    logger.info(f"successfully loaded raw features, shape: {df_x.shape}")
    logger.info(f"successfully loaded raw targets, shape: {df_y.shape}")
    
    id_col = config.schema.id_col
    
    # assert baseline column presence
    assert id_col in df_x.columns, f"feature dataset missing critical identifier column: '{id_col}'"
    assert id_col in df_y.columns, f"target dataset missing critical identifier column: '{id_col}'"
    assert config.schema.target_col in df_y.columns, f"target dataset missing expected target column: '{config.schema.target_col}'"
    
    # constraint check: ID uniqueness (prevents row inflation if duplicate ids exist)
    if not df_x[id_col].is_unique:
        logger.critical(f"integrity check failed: non-unique IDs found within raw features ('{id_col}')!")
        raise ValueError("raw feature IDs contains duplicate keys, ingestion halted to prevent row inflation.")
        
    if not df_y[id_col].is_unique:
        logger.critical(f"integrity check failed: non-unique IDs found within raw targets ('{id_col}')!")
        raise ValueError("raw target IDs contains duplicate keys, ingestion halted to prevent row inflation.")
        
    logger.info("uniqueness checks passed, no duplicate sequence keys identified.")
    
    # execute merge
    df_merged = pd.merge(df_x, df_y, on=id_col, how="inner")
    
    # post-merge check: row conservation
    expected_rows = len(df_x)
    actual_rows = len(df_merged)
    
    if actual_rows != expected_rows:
        logger.critical(
            f"row preservation constraint violated, expected rows: {expected_rows}, "
            f"actual merged rows: {actual_rows}."
        )
        raise AssertionError("inner join operations altered the dataset cardinality, check for unmatched records.")
        
    logger.info(f"data merge operation complete, conservation verified successfully, final shape: {df_merged.shape}")
    return df_merged