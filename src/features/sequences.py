import numpy as np
import pandas as pd
from src.utils.config import PipelineConfig
from src.utils.logger import setup_logger

logger = setup_logger()

def build_gap_aware_sequences(
    X_df: pd.DataFrame, 
    y_scaled: np.ndarray, 
    df_original: pd.DataFrame, 
    config: PipelineConfig
) -> tuple[np.ndarray, np.ndarray]:
    """
    transforms scaled 2D features and scaled 1D targets into 3D LSTM-compatible 
    tensors strictly within contiguous temporal blocks, avoiding gaps and 
    communication outages.
    """
    time_col = config.schema.time_col
    lookback = config.sequence_parameters.lookback_steps
    step_size = config.sequence_parameters.step_size
    
    # identify one-hot columns corresponding to turbines
    turbine_cols = [col for col in X_df.columns if "MAC_CODE" in col]
    
    X_seq, y_seq = [], []
    
    # align original metadata (Date_time) with scaled data index
    df_meta = df_original.loc[X_df.index]
    
    logger.info(f"generating 3D tensors, expected contiguous step delta: {step_size}")
    
    for col in turbine_cols:
        # isolate rows belonging strictly to this turbine
        mask = X_df[col] == 1.0
        if not mask.any():
            continue
            
        X_sub = X_df[mask].values
        y_sub = y_scaled[mask]
        times_sub = df_meta[mask][time_col].values
        
        # calculate sequential time difference to identify gaps
        deltas = np.diff(times_sub)
        gaps = deltas > step_size
        
        # generate contiguous block IDs based on cumulative sums of gaps
        gaps_extended = np.insert(gaps, 0, False)
        block_ids = np.cumsum(gaps_extended)
        
        # extract sliding sequences strictly within each block
        for block_id in np.unique(block_ids):
            block_mask = block_ids == block_id
            if np.sum(block_mask) <= lookback:
                continue
                
            X_block = X_sub[block_mask]
            y_block = y_sub[block_mask]
            
            for i in range(len(X_block) - lookback):
                X_seq.append(X_block[i : i + lookback])
                y_seq.append(y_block[i + lookback])
                
    X_seq_arr = np.array(X_seq)
    y_seq_arr = np.array(y_seq)
    
    logger.info(f"sequence construction complete.")
    logger.info(f"  -> generated 3D features shape: {X_seq_arr.shape}")
    logger.info(f"  -> generated 1D targets shape:  {y_seq_arr.shape}")
    return X_seq_arr, y_seq_arr