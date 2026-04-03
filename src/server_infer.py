# 서버단 CPU/GPU 추론 스크립트

from pathlib import Path
import numpy as np
import onnxruntime as ort

input_path = "/workspace/data/inputs_raw_fmt/simple_cnn/input_0.raw"
onnx_path = "/workspace/models/onnx/simple_cnn.onnx"
save_path = "/workspace/data/outputs/server_out/simple_cnn/output.raw"

x = np.fromfile(input_path, dtype=np.float32).reshape(1, 3, 224, 224)

sess = ort.InferenceSession(str(onnx_path))
ort_out = sess.run(None, {"input": x})[0]

print("input shape :", x.shape)
print("output shape:", ort_out.shape)
print("min/max     :", ort_out.min(), ort_out.max())

ort_out.astype(np.float32).tofile(save_path)
print("saved host output to:", save_path)