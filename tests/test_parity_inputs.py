import os
import numpy as np

def test_inputs_shape():
    path = os.path.join("artifacts", "golden", "inputs.npy")
    if not os.path.exists(path):
        # allow empty in CI; user runs parity_dump first
        return
    x = np.load(path)
    assert x.ndim == 4, x.shape  # (N,T,L,C)
    assert x.shape[1] == 100
    assert x.shape[2] == 20
