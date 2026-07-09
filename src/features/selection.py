import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from src.utils.config import PipelineConfig
from src.utils.logger import setup_logger

logger = setup_logger()

def compute_mi_scores(df: pd.DataFrame, config: PipelineConfig) -> pd.Series:
    """
    computes MI scores on a representative randomized sample 
    of the dataset, returns a pandas Series sorted by score.
    """
    id_col = config.schema.id_col
    target_col = config.schema.target_col
    time_col = config.schema.time_col
    turbine_col = config.schema.turbine_col
    
    excluded_cols = [id_col, target_col, time_col, turbine_col]
    feature_candidates = [col for col in df.columns if col not in excluded_cols]
    numerical_candidates = [col for col in feature_candidates if pd.api.types.is_numeric_dtype(df[col])]
    
    sample_size = min(config.feature_selection.mi_sample_size, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    
    X_sample = df_sample[numerical_candidates].copy()
    y_sample = df_sample[target_col]
    
    # impute NaNs using median to prevent k-NN crashes (temporary)
    for col in X_sample.columns:
        if X_sample[col].isnull().any():
            X_sample[col] = X_sample[col].fillna(X_sample[col].median())
            
    mi_scores = mutual_info_regression(X_sample, y_sample, random_state=42)
    mi_series = pd.Series(mi_scores, index=numerical_candidates).sort_values(ascending=False)
    return mi_series


def identify_low_relevance_features(mi_series: pd.Series, threshold: float) -> list[str]:
    """filters out features falling below the information threshold."""
    low_relevance = list(mi_series[mi_series < threshold].index)
    logger.info("=== top 5 most informative features ===")
    for col, score in mi_series.head(5).items():
        logger.info(f"  -> {col:<40} : {score:.4f} bits")
        
    logger.info("=== bottom 5 least informative features ===")
    for col, score in mi_series.tail(5).items():
        logger.info(f"  -> {col:<40} : {score:.4f} bits")
        
    logger.info(f"identified {len(low_relevance)} columns with MI < {threshold} bits.")
    return low_relevance


def identify_collinear_features(df: pd.DataFrame, config: PipelineConfig, mi_series: pd.Series, excluded_features: list[str]) -> list[str]:
    """
    executes agglomerative hierarchical clustering on pairwise Spearman rank correlation matrices. 
	for each cluster of redundant variables, retains the feature with the highest MI score, 
	flagging others for pruning.
    """
    id_col = config.schema.id_col
    target_col = config.schema.target_col
    time_col = config.schema.time_col
    turbine_col = config.schema.turbine_col
    
    # exclude metadata, target, and low-relevance features
    system_excludes = [id_col, target_col, time_col, turbine_col] + excluded_features
    candidates = [col for col in df.columns if col not in system_excludes]
    numerical_candidates = [col for col in candidates if pd.api.types.is_numeric_dtype(df[col])]
    
    # sample the features for faster rank correlation computing (same sample size)
    sample_size = min(config.feature_selection.mi_sample_size, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    
    X_sample = df_sample[numerical_candidates].copy()
    for col in X_sample.columns:
        if X_sample[col].isnull().any():
            X_sample[col] = X_sample[col].fillna(X_sample[col].median())
            
    logger.info("computing pairwise Spearman rank correlation matrix...")
    corr_matrix = X_sample.corr(method="spearman")
    
    # convert correlation matrix to distance matrix: D = 1 - |r_s|
    dist_matrix = (1.0 - corr_matrix.abs()).clip(0.0, 1.0)
    
    # convert to condensed 1D distance vector
    condensed_dist = squareform(dist_matrix, force="tovector", checks=False)
    
    # perform complete linkage clustering
    logger.info("performing complete-linkage hierarchical clustering...")
    Z = linkage(condensed_dist, method="complete")
    
    # form flat clusters using distance threshold (1 - threshold)
    threshold_dist = 1.0 - config.feature_selection.spearman_threshold
    cluster_labels = fcluster(Z, t=threshold_dist, criterion="distance")
    
    # map features to clusters
    cluster_groups = {}
    for col, label in zip(corr_matrix.columns, cluster_labels):
        cluster_groups.setdefault(label, []).append(col)
        
    cols_to_prune = []
    logger.info(f"=== multicollinearity pruning (threshold: r_s > {config.feature_selection.spearman_threshold}) ===")
    
    for label, cols in sorted(cluster_groups.items()):
        if len(cols) > 1:
            # sort variables within this cluster by their MI score
            sorted_by_mi = sorted(cols, key=lambda x: mi_series.get(x, 0.0), reverse=True)
            best_feature = sorted_by_mi[0]
            redundant_features = sorted_by_mi[1:]
            
            cols_to_prune.extend(redundant_features)
            logger.info(
                f"  -> cluster {label}: keeping '{best_feature}' (MI: {mi_series.get(best_feature, 0.0):.4f}). "
                f"pruning redundant: {redundant_features}"
            )
            
    logger.info(f"identified {len(cols_to_prune)} redundant collinear columns.")
    return cols_to_prune


def run_feature_selection_pipeline(df: pd.DataFrame, config: PipelineConfig) -> list[str]:
    """
    runs the complete feature selection pipeline.
    returns a unified list of columns that should be dropped from the dataset.
    """
    # compute mutual information relevance scores
    mi_series = compute_mi_scores(df, config)
    
    # filter low-relevance features
    low_relevance_features = identify_low_relevance_features(
        mi_series, config.feature_selection.mi_threshold
    )
    
    # prune collinear features based on Spearman clustering
    collinear_features = identify_collinear_features(
        df, config, mi_series, excluded_features=low_relevance_features
    )
    
    unified_drop_list = low_relevance_features + collinear_features
    logger.info(f"feature selection complete, total features flagged for removal: {len(unified_drop_list)}")
    return unified_drop_list