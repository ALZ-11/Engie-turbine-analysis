import numpy as np
import pandas as pd
from src.utils.config import PipelineConfig
from src.utils.logger import setup_logger

logger = setup_logger()

def split_chronologically(
    df: pd.DataFrame, 
    config: PipelineConfig, 
    train_ratio: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    splits a dataframe chronologically based on the unique values of the timestamp column.
    guarantees that records sharing the same timestamp across multiple turbines 
    are grouped together, preventing spatiotemporal data leakage.
    """
    time_col = config.schema.time_col
    
    # extract the unique timestamp values and sort them
    unique_times = np.sort(df[time_col].unique())
    
    # determine the time boundary corresponding to the split percentile
    split_idx = int(len(unique_times) * train_ratio)
    split_time_boundary = unique_times[split_idx]
    
    # partition the dataset strictly by the value of the timestamp boundary
    df_train = df[df[time_col] < split_time_boundary]
    df_test = df[df[time_col] >= split_time_boundary]
    
    # log rigorous split analytics for verification
    total_rows = len(df)
    logger.info("=== Chronological Value-Based Partitioning ===")
    logger.info(f"Split Time Boundary   : {split_time_boundary}")
    logger.info(f"Training Partition    : {len(df_train)} rows ({len(df_train)/total_rows*100:.2f}%)")
    logger.info(f"                      -> Timeframe: {df_train[time_col].min()} to {df_train[time_col].max()}")
    logger.info(f"Testing Partition     : {len(df_test)} rows ({len(df_test)/total_rows*100:.2f}%)")
    logger.info(f"                      -> Timeframe: {df_test[time_col].min()} to {df_test[time_col].max()}")
    
    # assert no overlap or leak
    assert df_train[time_col].max() < df_test[time_col].min(), \
        "CRITICAL: data leakage identified, training and testing timeframe partitions overlap."
        
    logger.info("validation split is verified as clean and chronologically contiguous.")
    return df_train, df_test