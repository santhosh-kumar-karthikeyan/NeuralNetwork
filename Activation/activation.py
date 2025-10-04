def bipolar(input: float, threshold: float) -> int:
    return 1 if input > threshold else -1

def binary(input: float, threshold: float) -> int:
    return 1 if input > threshold else 0