import numpy as np

def binary(input_val: float, threshold: float = 0) -> int:
    """Binary step function: returns 0 or 1."""
    return 1 if input_val > threshold else 0

def bipolar(input_val: float, threshold: float = 0) -> int:
    """Bipolar step function: returns -1 or 1."""
    return 1 if input_val > threshold else -1

def step(x: float, threshold: float = 0, mode: str = "binary") -> int:
    """
    Step function that supports both binary and bipolar modes.
    
    Args:
        x: Input value
        threshold: Activation threshold
        mode: "binary" (0/1) or "bipolar" (-1/1)
    
    Returns:
        1 or 0 (binary mode) or 1 or -1 (bipolar mode)
    """
    if mode == "bipolar":
        return bipolar(x, threshold)
    else:
        return binary(x, threshold)

def sigmoid(x: float, mode: str = "binary") -> float:
    """
    Sigmoid activation function.
    
    Args:
        x: Input value
        mode: "binary" or "bipolar" (affects scaling)
    
    Returns:
        Sigmoid output scaled appropriately for the mode
    """
    sig = 1 / (1 + np.exp(-x))
    if mode == "bipolar":
        # Scale from [0, 1] to [-1, 1]
        return 2 * sig - 1
    else:
        # Keep in [0, 1] for binary mode
        return sig

def threshold_fn(x: float, threshold: float = 0, mode: str = "binary") -> int:
    """
    Threshold/step activation function (alias for step).
    
    Args:
        x: Input value
        threshold: Activation threshold
        mode: "binary" (0/1) or "bipolar" (-1/1)
    
    Returns:
        Step function output
    """
    return step(x, threshold, mode)
