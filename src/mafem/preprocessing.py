import numpy as np

def low_pass_filter(signal, alpha=0.1):
    """
    Exponential moving average low-pass filter.
    Formula: y(t) = alpha * x(t) + (1 - alpha) * y(t-1)
    """
    if len(signal) == 0:
        return np.array([], dtype=float)
    
    filtered = np.zeros_like(signal, dtype=float)
    filtered[0] = signal[0]
    for t in range(1, len(signal)):
        filtered[t] = alpha * signal[t] + (1 - alpha) * filtered[t - 1]
    return filtered

def min_max_normalize(data):
    """
    Min-Max normalization to scale values to the range [0, 1].
    """
    if len(data) == 0:
        return data
        
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    denom = max_val - min_val
    
    # Avoid division by zero
    if isinstance(denom, np.ndarray):
        denom[denom == 0] = 1.0
    else:
        if denom == 0:
            denom = 1.0
            
    return (data - min_val) / denom
