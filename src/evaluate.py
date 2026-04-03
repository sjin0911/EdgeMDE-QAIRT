# 출력 텐서 간의 MSE, Cosine Similarity 등 오차 검증 로직

from pathlib import Path
import numpy as np

def mse(a, b):
    return np.mean((a - b) ** 2)

host_output_path = "/workspace/data/outputs/server_out/simple_cnn/output.raw"
cpu_output_path = "/workspace/data/outputs/cpu_out/simple_cnn/output/Result_0/output.raw"
npu_output_path = "/workspace/data/outputs/npu_out/simple_cnn/output_dsp/Result_0/output.raw"
shape = (1, 1, 56, 56)

host_out = np.fromfile(host_output_path, dtype=np.float32).reshape(shape)
cpu_out = np.fromfile(cpu_output_path, dtype=np.float32).reshape(shape)
npu_out = np.fromfile(npu_output_path, dtype=np.float32).reshape(shape)

print("host shape       :", host_out.shape)
print("device cpu shape :", cpu_out.shape)
print("device npu shape :", npu_out.shape)
print()

print("host min/max       :", host_out.min(), host_out.max())
print("device cpu min/max :", cpu_out.min(), cpu_out.max())
print("device npu min/max :", npu_out.min(), npu_out.max())
print()

print("[host vs. cpu]")
print("MSE      :", mse(host_out, cpu_out))
print("MAE      :", np.mean(np.abs(host_out - cpu_out)))
print("MAX_ABS  :", np.max(np.abs(host_out - cpu_out)))
print()

print("[host vs. npu]")
print("MSE      :", mse(host_out, npu_out))
print("MAE      :", np.mean(np.abs(host_out - npu_out)))
print("MAX_ABS  :", np.max(np.abs(host_out - npu_out)))
print()

print("[cpu vs. npu]")
print("MSE      :", mse(cpu_out, npu_out))
print("MAE      :", np.mean(np.abs(cpu_out - npu_out)))
print("MAX_ABS  :", np.max(np.abs(cpu_out - npu_out)))