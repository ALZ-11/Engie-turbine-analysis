import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_evaluation_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    standby_min_kw: float
) -> dict[str, float]:
    """
    computes standard statistical metrics (MAE, RMSE, R2) and domain-specific 
    physical boundary violations in original physical units (kW).
    """
    # force convert to float64 arrays (avoid numerical underflow)
    y_true_64 = np.array(y_true, dtype=np.float64)
    y_pred_64 = np.array(y_pred, dtype=np.float64)
    
    mae = mean_absolute_error(y_true_64, y_pred_64)
    rmse = np.sqrt(mean_squared_error(y_true_64, y_pred_64))
    r2 = r2_score(y_true_64, y_pred_64)
    
    # calculate proportion of predictions violating raw physical boundary limits
    floor_violations = np.sum(y_pred_64 < standby_min_kw)
    total_samples = len(y_pred_64)
    violation_rate = (floor_violations / total_samples) * 100.0
    
    return {
        "MAE (kW)": float(mae),
        "RMSE (kW)": float(rmse),
        "R2 Score": float(r2),
        "Floor Violations": int(floor_violations),
        "Floor Violation Rate (%)": float(violation_rate)
    }