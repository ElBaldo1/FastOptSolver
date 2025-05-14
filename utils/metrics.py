import numpy as np

def mse(y_true, y_pred):
    """Calculate Mean Squared Error (MSE) between true and predicted values.
    
    Args:
        y_true: Array-like of true target values
        y_pred: Array-like of predicted values
        
    Returns:
        float: Mean squared error
        
    Raises:
        ValueError: If inputs have different shapes or are empty
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError("Input arrays must have the same shape")
    if y_true.size == 0:
        raise ValueError("Input arrays cannot be empty")
        
    return np.mean((y_true - y_pred) ** 2)

def mae(y_true, y_pred):
    """Calculate Mean Absolute Error (MAE) between true and predicted values.
    
    Args:
        y_true: Array-like of true target values
        y_pred: Array-like of predicted values
        
    Returns:
        float: Mean absolute error
        
    Raises:
        ValueError: If inputs have different shapes or are empty
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError("Input arrays must have the same shape")
    if y_true.size == 0:
        raise ValueError("Input arrays cannot be empty")
        
    return np.mean(np.abs(y_true - y_pred))

def r2(y_true, y_pred):
    """Calculate R-squared (coefficient of determination) score.
    
    Args:
        y_true: Array-like of true target values
        y_pred: Array-like of predicted values
        
    Returns:
        float: R-squared score (can range from -inf to 1)
        
    Raises:
        ValueError: If inputs have different shapes or are empty
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError("Input arrays must have the same shape")
    if y_true.size == 0:
        raise ValueError("Input arrays cannot be empty")
        
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 0.0  # All values are the same
    
    return 1 - (ss_res / ss_tot)