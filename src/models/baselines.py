import numpy as np

class PersistenceBaselineModel:
    """
    non-parametric persistence estimator, assumes power generated at 
    time t is identical to power observed at time t-1.
    """
    def __init__(self) -> None:
        pass

    def predict(self, y_true: np.ndarray, last_known_train_value: float = None) -> np.ndarray:
        """
        generates persistence forecasts by rolling the true targets array by 1.
        optionally uses the last value of the training timeframe to ensure zero 
        cross-boundary leakage.
        """
        y_true_arr = np.array(y_true, dtype=np.float64)
        
        # roll targets right by 1 step
        y_pred = np.roll(y_true_arr, shift=1)
        
        # resolve boundary condition
        if last_known_train_value is not None:
            y_pred[0] = float(last_known_train_value)
        else:
            y_pred[0] = y_true_arr[0] # fallback to first test point if train data is missing
            
        return y_pred