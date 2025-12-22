import os
import numpy as np

def test_outputs_exist_and_shape():
    base = os.path.join("artifacts", "golden")
    if not os.path.exists(os.path.join(base, "inputs.npy")):
        return
    for h in (10, 50, 100):
        p = os.path.join(base, f"logits_{h}.npy")
        assert os.path.exists(p), p
        y = np.load(p)
        assert y.ndim == 2 and y.shape[1] == 3, y.shape
