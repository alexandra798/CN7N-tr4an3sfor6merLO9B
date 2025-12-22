"""Very small smoke test: measures onnxruntime latency (Python).

Run:
  python tests/latency_smoke.py
"""
import os, time
import numpy as np
import onnxruntime as ort

def main():
    model_path = os.path.join("artifacts", "model.onnx")
    if not os.path.exists(model_path):
        print("missing artifacts/model.onnx (export first)")
        return
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    x = np.zeros((1,100,20,7), dtype=np.float32)
    # warmup
    for _ in range(10):
        _ = sess.run(None, {"input_lob": x})
    t0 = time.perf_counter()
    N = 200
    for _ in range(N):
        _ = sess.run(None, {"input_lob": x})
    dt = (time.perf_counter() - t0) / N * 1e3
    print(f"avg latency: {dt:.3f} ms")

if __name__ == "__main__":
    main()
