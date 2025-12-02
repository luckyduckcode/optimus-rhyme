import pytest
import numpy as np
try:
    import q4_kernel
except ImportError:
    pytest.skip("q4_kernel module not installed", allow_module_level=True)

def test_pack_unpack_roundtrip():
    # Generate random data in range [0, 15]
    n = 1024
    original = np.random.randint(0, 16, size=n, dtype=np.uint8)
    
    # Pack
    packed_bytes = q4_kernel.pack_q4(original.tobytes())
    assert len(packed_bytes) == n // 2
    
    # Unpack
    unpacked_bytes = q4_kernel.unpack_q4(packed_bytes, n)
    unpacked = np.frombuffer(unpacked_bytes, dtype=np.uint8)
    
    # Verify
    np.testing.assert_array_equal(original, unpacked)

def test_pack_unpack_odd_size_error():
    data = b'\x01\x02\x03' # 3 bytes
    with pytest.raises(RuntimeError):
        q4_kernel.pack_q4(data)

def test_unpack_small_buffer_error():
    packed = b'\x01' # 1 byte -> 2 elements
    with pytest.raises(RuntimeError):
        q4_kernel.unpack_q4(packed, 10) # Expecting 10 elements (5 bytes)
