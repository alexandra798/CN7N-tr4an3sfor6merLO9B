import os
import numpy as np

def _max_rel_err(a, b, eps=1e-12):
    denom = np.maximum(np.abs(b), eps)
    return np.max(np.abs(a-b) / denom)

def test_cpp_vs_py_parity_if_present():
    base = os.path.join("artifacts", "golden")
    cpp = os.path.join(base, "cpp_out")
    if not os.path.exists(os.path.join(base, "logits_10.npy")):
        return
    if not os.path.exists(os.path.join(cpp, "logits_10.npy")):
        # user didn't run C++ parity executable
        return

    for h in (10, 50, 100):
        py = np.load(os.path.join(base, f"logits_{h}.npy")).astype(np.float32)
        cc = np.load(os.path.join(cpp, f"logits_{h}.npy")).astype(np.float32)
        assert py.shape == cc.shape
        err = _max_rel_err(cc, py)
        # loose by default; tighten after you ensure same ORT options / seeds
        assert err < 1e-3, f"h={h} rel_err={err}"
