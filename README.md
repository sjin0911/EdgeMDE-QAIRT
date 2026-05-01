# Depth Anything V2 on Galaxy S20 DSP with Qualcomm QAIRT

This repository documents an end-to-end deployment and debugging study for running **Depth Anything V2 ViT-S** on a **Samsung Galaxy S20 (SM8250)** using **Qualcomm AI Runtime (QAIRT / QNN)**.

The current milestone is no longer a dummy `SimpleCNN` pipeline check. The project now contains a working Depth Anything V2 deployment path that reaches real on-device inference on the Galaxy S20 DSP and produces a depth map output.

The final successful artifact is:

```text
models/dlc/depth_anything_v2_vits_518_dsp_gatherfix_quick_gelu_quantized_bias8.dlc
```

This DLC is a DSP-compatible approximation of the original Depth Anything V2 model. The original PyTorch checkpoint is preserved, but the exported graph requires several targeted changes to satisfy the SM8250 DSP v66 backend constraints.

## Executive Summary

- **Model:** Depth Anything V2 ViT-S
- **Checkpoint:** `models/pytorch/depth_anything_v2_vits.pth`
- **Input resolution:** `1 x 3 x 518 x 518`
- **Target device:** Samsung Galaxy S20, SM8250
- **Validated accelerator path:** QNN DSP v66
- **Final backend library:** `libQnnDsp.so`
- **Final DLC:** `depth_anything_v2_vits_518_dsp_gatherfix_quick_gelu_quantized_bias8.dlc`
- **Final device output:** `data/outputs/device_dsp/depth_anything_v2_vits_quick_gelu_bias8/output_quick_gelu_bias8/Result_0/output.raw`

The final QuickGELU DSP output is visually close to the original CPU depth structure and numerically much better than the earlier ReLU and SiLU DSP-compatible trials.

![QuickGELU DSP depth map](data/outputs/device_dsp/depth_anything_v2_vits_quick_gelu_bias8/dechirico_depth_device_dsp_quick_gelu_bias8.png)

## Target Backend: DSP, Not HTP

The Galaxy S20 used in this project is based on SM8250. Runtime validation showed:

```text
Backend      : DSP
Core Version : Hexagon Architecture V66
Unit Test    : Passed
```

Therefore, the correct execution path for this device is the **QNN DSP v66 path**, not the newer HTP v68 path.

The device-side runtime uses:

```text
libQnnDsp.so
libQnnDspV66Stub.so
libQnnModelDlc.so
libQnnSystem.so
dsp/libQnnDspV66.so
dsp/libQnnDspV66Skel.so
```

The required runtime environment on device is:

```bash
export LD_LIBRARY_PATH=/data/local/tmp/depth_anything_v2_vits_dsp
export ADSP_LIBRARY_PATH="/data/local/tmp/depth_anything_v2_vits_dsp/dsp;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp"
```

Some QAIRT tools still print HTP-related warnings because the SDK's quantizer defaults to HTP-aware logic. The actual on-device execution path validated here is DSP v66.

## Why the Original Model Did Not Run Directly

Depth Anything V2 is not a simple convolutional network. Its DINOv2 encoder contains transformer attention, LayerNorm, Softmax, MLP activation functions, tensor reshapes, and interpolation. Several of these patterns were rejected by the SM8250 DSP backend.

The table below summarizes the critical blockers and the implemented fixes.

| Stage | Original behavior | DSP failure | Implemented change |
| --- | --- | --- | --- |
| Attention Q/K/V split | Rank-5 `qkv` tensor indexed by `qkv[0]`, `qkv[1]`, `qkv[2]` | DSP rejected rank-5 `Gather` | Split Q/K/V while tensor is still rank-3, then reshape each branch |
| LayerNorm quantization | LayerNorm bias quantized to `sFxp_32` when using 32-bit bias | DSP rejected LayerNorm input datatype | Quantize with `--bias_bitwidth 8` |
| MLP activation | DINOv2 uses `nn.GELU` | QAIRT lowered GELU to `ElementWiseNeuron operation: 1`, rejected by DSP v66 | Replace GELU with DSP-compatible `QuickGELU` approximation |

The full debugging history is recorded in:

```text
docs/depth_anything_v2_sm8250_dsp_debug_log.md
```

## Core Model Changes

### 1. Attention Export Rewrite

The original DINOv2 attention implementation creates a rank-5 QKV tensor and indexes it:

```python
qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
```

ONNX export converted this into `Gather` operations over a rank-5 tensor. The SM8250 DSP v66 validator rejected this pattern:

```text
/model/blocks.0/attn/Gather:qti.aisw:Gather
Input[0] has incorrect Rank 5
```

The implementation now splits Q/K/V before increasing tensor rank:

```python
head_dim = C // self.num_heads
q, k, v = self.qkv(x).chunk(3, dim=-1)
q = q.reshape(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)
k = k.reshape(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)
v = v.reshape(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)
q = q * self.scale
```

This preserves the same mathematical attention computation while producing a DSP-compatible export pattern.

Relevant file:

```text
external/Depth-Anything-V2/depth_anything_v2/dinov2_layers/attention.py
```

### 2. 8-bit Bias Quantization for LayerNorm

After fixing attention, the next DSP blocker appeared at LayerNorm:

```text
_layernorm_0:qti.aisw:LayerNorm
Input[2] has incorrect Datatype 0x332
```

The problematic tensor was the LayerNorm bias encoded as `sFxp_32`. Quantizing with 8-bit bias avoided this unsupported datatype:

```bash
--act_bitwidth 8 \
--weights_bitwidth 8 \
--bias_bitwidth 8 \
--use_per_channel_quantization
```

This setting is required for the final DSP DLC.

### 3. QuickGELU Instead of Original GELU

The original Depth Anything V2 DINOv2 MLP uses GELU. QAIRT lowered this to:

```text
ElementWiseNeuron operation: 1
```

SM8250 DSP v66 rejected that operation parameter. ReLU and SiLU were tested as alternatives, but ReLU lost too much depth structure and SiLU produced mostly edge-like responses.

The best current activation replacement is:

```python
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)
```

QuickGELU exports as `Sigmoid` plus elementwise multiplication, which the DSP accepts, while remaining much closer to GELU than ReLU.

Relevant files:

```text
external/Depth-Anything-V2/depth_anything_v2/dinov2.py
external/Depth-Anything-V2/depth_anything_v2/dpt.py
src/models/depth_anything_v2_vits.py
configs/depth_model_dsp_quick_gelu.yaml
```

## Visual Results

The following image compares four outputs from the same input. From left to right:

1. Original Depth Anything V2 on CPU
2. DSP-compatible ReLU variant
3. DSP-compatible SiLU variant
4. DSP-compatible QuickGELU variant

![Original CPU vs DSP activation variants](data/outputs/device_dsp/depth_anything_v2_vits_quick_gelu_bias8/compare_cpu_original_relu_silu_quick_gelu.png)

The QuickGELU output preserves the global foreground/background depth structure much better than the ReLU and SiLU trials. It is not identical to the original model, but it is the best DSP-compatible approximation produced in this debugging pass.

## Numerical Evaluation

All comparisons below use the same `518 x 518` float32 output tensor. The original host CPU result is the unmodified Depth Anything V2 float DLC executed with the QNN CPU backend on the host. The QuickGELU results use the final quantized DLC.

### Pairwise Output Agreement

| Comparison | Pixelwise MSE | MAE | Max Abs | Pearson Corr. |
| --- | ---: | ---: | ---: | ---: |
| Original host CPU vs QuickGELU host CPU | 1.7039 | 1.0319 | 3.7646 | 0.9672 |
| Original host CPU vs QuickGELU device CPU | 1.5704 | 1.0116 | 3.7159 | 0.9652 |
| Original host CPU vs QuickGELU device DSP | 1.6529 | 1.0492 | 3.8702 | 0.9585 |
| QuickGELU host CPU vs QuickGELU device CPU | 0.008756 | 0.07727 | 1.5524 | 0.9981 |
| QuickGELU host CPU vs QuickGELU device DSP | 0.007933 | 0.07473 | 1.4091 | 0.9971 |
| QuickGELU device CPU vs QuickGELU device DSP | 0.007913 | 0.06065 | 1.6002 | 0.9957 |

The most important observations are:

- QuickGELU changes the original model behavior, but remains strongly correlated with the original CPU output.
- The final DLC is highly consistent across host CPU, device CPU, and device DSP.
- The device DSP output is not a random or collapsed tensor; it tracks the same depth structure with high correlation.

### Activation Variant Comparison

| DSP-compatible activation | DSP execution | Pixelwise MSE vs original host CPU | MAE vs original host CPU | Pearson Corr. vs original host CPU | Qualitative result |
| --- | --- | ---: | ---: | ---: | --- |
| Original GELU | Failed | N/A | N/A | N/A | Rejected as `ElementWiseNeuron operation: 1` |
| ReLU | Succeeded | 3.2756 | 1.4746 | 0.8583 | Over-smoothed and compressed depth range |
| SiLU | Succeeded | 6.4730 | 1.7613 | 0.2691 | Mostly edge-like response |
| QuickGELU | Succeeded | 1.6529 | 1.0492 | 0.9585 | Best current DSP-compatible approximation |

## End-to-End Runtime Measurement

The following timings were measured on the final QuickGELU quantized DLC with the same host-side `time.perf_counter()` wrapper. Every run produced a valid `output.raw` tensor of `1,073,296` bytes. These are **end-to-end wall-clock measurements** and include model loading, graph composition/finalization, host-device command overhead for device runs, inference execution, and output writing. They should not be interpreted as pure accelerator kernel latency.

| Runtime target | Backend library | Device/environment | Cold single-run times | Mean cold time |
| --- | --- | --- | ---: | ---: |
| Host CPU | `libQnnCpu.so` | Docker container, x86_64 host | 0.762 s, 0.742 s | 0.752 s |
| Device CPU | `libQnnCpu.so` | Galaxy S20, ARM CPU | 1.667 s, 1.683 s | 1.675 s |
| Device DSP | `libQnnDsp.so` | Galaxy S20, Hexagon DSP v66 | 106.842 s, 106.826 s | 106.834 s |

To check whether the DSP time was dominated by one-time graph preparation, an additional `--num_inferences 2 --keep_num_outputs 1` run was performed:

| Runtime target | Total time for 2 inferences | Amortized time per inference |
| --- | ---: | ---: |
| Host CPU | 1.451 s | 0.725 s |
| Device CPU | 3.134 s | 1.567 s |
| Device DSP | 212.709 s | 106.354 s |

The DSP result is currently a **correctness milestone**, not a latency-optimized deployment. Although DSPs are usually expected to outperform CPUs for well-supported quantized neural-network kernels, that expectation does not automatically hold for this first full-resolution transformer graph on SM8250 DSP v66.

The two-inference measurement shows that the 106-second DSP result is not merely a one-time initialization artifact; the execution path remains approximately 106 seconds per inference after amortization. The measured DSP path is slow because the graph is dominated by transformer-style operations such as MatMul, Softmax, LayerNorm, Slice/Reshape/Transpose, and large attention tensors rather than a mobile-optimized convolutional workload. In addition, this run uses direct DLC execution without cached context binaries, per-op profiling-guided optimization, model partitioning, or resolution reduction. Therefore, the current DSP number should be read as proof that the graph can execute on DSP, not as evidence of an optimized mobile inference latency.

Future performance work should focus on:

- generating and loading a cached QNN context binary;
- profiling per-op execution on DSP;
- reducing unsupported or expensive transformer patterns;
- testing lower input resolutions that remain multiples of the ViT patch size;
- improving calibration data beyond the current single-image PTQ setup;
- considering architectural distillation into a mobile-native depth model.

## Reproducing the Current Best DSP Result

### 1. Export QuickGELU ONNX

```bash
PYTHONPATH=/workspace python3 /workspace/src/export_onnx.py \
  --config /workspace/configs/depth_model_dsp_quick_gelu.yaml
```

Expected output:

```text
models/onnx/depth_anything_v2_vits_518_dsp_gatherfix_quick_gelu.onnx
```

### 2. Convert ONNX to DLC

```bash
source /workspace/qairt/2.45.0.260326/bin/envsetup.sh

qairt-converter \
  --input_network /workspace/models/onnx/depth_anything_v2_vits_518_dsp_gatherfix_quick_gelu.onnx \
  --source_model_input_shape input 1,3,518,518 \
  --target_backend DSP \
  --output_path /workspace/models/dlc/depth_anything_v2_vits_518_dsp_gatherfix_quick_gelu_float.dlc
```

### 3. Quantize for DSP

```bash
qairt-quantizer \
  --input_dlc /workspace/models/dlc/depth_anything_v2_vits_518_dsp_gatherfix_quick_gelu_float.dlc \
  --input_list /workspace/data/inputs_raw_fmt/depth_anything_v2_vits/input_list_host.txt \
  --output_dlc /workspace/models/dlc/depth_anything_v2_vits_518_dsp_gatherfix_quick_gelu_quantized_bias8.dlc \
  --act_bitwidth 8 \
  --weights_bitwidth 8 \
  --bias_bitwidth 8 \
  --use_per_channel_quantization
```

### 4. Run on Galaxy S20 DSP

```bash
TARGET=/data/local/tmp/depth_anything_v2_vits_dsp

adb push /workspace/models/dlc/depth_anything_v2_vits_518_dsp_gatherfix_quick_gelu_quantized_bias8.dlc $TARGET/

adb shell "cd $TARGET && \
  rm -rf output_quick_gelu_bias8 && mkdir -p output_quick_gelu_bias8 && \
  export LD_LIBRARY_PATH=$TARGET && \
  export ADSP_LIBRARY_PATH='$TARGET/dsp;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp' && \
  ./qnn-net-run \
    --backend libQnnDsp.so \
    --model libQnnModelDlc.so \
    --dlc_path depth_anything_v2_vits_518_dsp_gatherfix_quick_gelu_quantized_bias8.dlc \
    --input_list input_list_target_dsp.txt \
    --output_dir output_quick_gelu_bias8 \
    --log_level verbose"
```

### 5. Pull Output

```bash
mkdir -p /workspace/data/outputs/device_dsp/depth_anything_v2_vits_quick_gelu_bias8

adb pull \
  /data/local/tmp/depth_anything_v2_vits_dsp/output_quick_gelu_bias8 \
  /workspace/data/outputs/device_dsp/depth_anything_v2_vits_quick_gelu_bias8/
```

The final raw tensor is saved as:

```text
data/outputs/device_dsp/depth_anything_v2_vits_quick_gelu_bias8/output_quick_gelu_bias8/Result_0/output.raw
```

The output is dequantized float32 because `qnn-net-run` was executed without `--use_native_output_files`.

## Repository Structure

```text
.
├── configs/
│   ├── depth_model_dsp.yaml
│   ├── depth_model_dsp_opset20.yaml
│   ├── depth_model_dsp_quick_gelu.yaml
│   ├── depth_model_dsp_relu.yaml
│   └── depth_model_dsp_silu.yaml
├── data/
│   ├── inputs_raw_fmt/depth_anything_v2_vits/
│   └── outputs/
│       ├── qnn_host_cpu/
│       ├── device_cpu/
│       └── device_dsp/
├── docs/
│   └── depth_anything_v2_sm8250_dsp_debug_log.md
├── external/
│   └── Depth-Anything-V2/
├── models/
│   ├── pytorch/depth_anything_v2_vits.pth
│   ├── onnx/
│   └── dlc/
├── qairt/
│   └── 2.45.0.260326/
├── src/
│   ├── export_onnx.py
│   └── models/depth_anything_v2_vits.py
└── README.md
```

## Current Limitations

This project has demonstrated real DSP execution, but it is not yet a production-quality mobile depth runtime.

Current limitations:

- The final model is not bit-exact to the original Depth Anything V2 because GELU is replaced with QuickGELU.
- Quantization calibration currently uses a very small input set, so accuracy may vary across scenes.
- The current DSP path is much slower than the CPU path in single-shot `qnn-net-run` timing.
- The output is relative depth, not metric depth.
- No Android application integration has been implemented yet.

## Engineering Significance

This work is valuable because it goes beyond a standard model export:

- it identifies backend-specific operator constraints on a real mobile SoC;
- it performs controlled graph-level surgery while tracking numerical consequences;
- it validates host, device CPU, and device DSP consistency;
- it records failure modes and fixes in a reproducible debug log;
- and it produces an actual depth map from the Galaxy S20 DSP path.

The main technical result is that Depth Anything V2 ViT-S can be brought onto SM8250 DSP v66 through targeted export rewrites, quantization constraints, and a GELU-compatible activation approximation.

## Affiliation

Efficient & Versatile AI Lab (EV Lab)  
Konkuk University
