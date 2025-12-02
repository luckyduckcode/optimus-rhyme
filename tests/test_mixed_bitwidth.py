import pytest
import numpy as np

# Mocking a mixed bitwidth scenario
# In a real scenario, this would involve a model with some layers in q4 and some in q8.
# Here we test that we can process data that is conceptually "mixed".

try:
    import q4_kernel
except ImportError:
    pytest.skip("q4_kernel module not installed", allow_module_level=True)

def test_mixed_bitwidth_processing():
    # Create a "tensor" that is half q4 and half q8 (uncompressed for this test)
    
    # Part 1: Q4 data
    n_q4 = 128
    data_q4 = np.random.randint(0, 16, size=n_q4, dtype=np.uint8)
    packed_q4 = q4_kernel.pack_q4(data_q4.tobytes())
    
    # Part 2: Q8 data (standard uint8)
    n_q8 = 128
    data_q8 = np.random.randint(0, 256, size=n_q8, dtype=np.uint8)
    
    # "Inference" step: Unpack Q4 and add to Q8 (element-wise, just as a dummy op)
    unpacked_q4_bytes = q4_kernel.unpack_q4(packed_q4, n_q4)
    unpacked_q4 = np.frombuffer(unpacked_q4_bytes, dtype=np.uint8)
    
    # Result
    result = unpacked_q4.astype(np.int16) + data_q8.astype(np.int16)
    
    # Verify logic
    expected = data_q4.astype(np.int16) + data_q8.astype(np.int16)
    np.testing.assert_array_equal(result, expected)
