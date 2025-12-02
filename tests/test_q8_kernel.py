import pytest
import numpy as np

def test_q8_flow():
    # Q8 is typically just 8-bit integers.
    # This test ensures that our environment handles 8-bit operations correctly.
    
    n = 1024
    data = np.random.randint(0, 256, size=n, dtype=np.uint8)
    
    # Simulate a "kernel" operation: scaling
    scale = 0.5
    result = (data * scale).astype(np.uint8)
    
    # Check bounds
    assert result.max() <= 255
    assert result.min() >= 0
    
    # Check a specific value
    idx = 0
    expected = int(data[idx] * scale)
    assert result[idx] == expected
