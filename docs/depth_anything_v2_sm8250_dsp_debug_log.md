# Depth Anything V2 on Galaxy S20 SM8250 DSP/HTP Debug Log

Date: 2026-05-01

Target:

- Device: Samsung Galaxy S20 SM-G981N
- SoC: SM8250, board/platform `kona`, ABI `arm64-v8a`
- SDK: QAIRT `/workspace/qairt/2.45.0.260326`
- Model: Depth Anything V2 ViT-S
- Source checkpoint: `/workspace/models/pytorch/depth_anything_v2_vits.pth`
- ONNX: `/workspace/models/onnx/depth_anything_v2_vits_518_sim.onnx`
- Float DLC: `/workspace/models/dlc/depth_anything_v2_vits_518_float.dlc`

## Baseline Already Verified

Before starting accelerator work, the model was verified on CPU:

- PyTorch to ONNX export passed.
- QAIRT float DLC conversion passed.
- `qnn-net-run` on Galaxy S20 CPU passed with `libQnnCpu.so`.
- ONNX vs Galaxy S20 CPU output:
  - MSE: `2.3038733161229175e-06`
  - MAE: `0.0010130195878446102`
  - MAX_ABS: `0.01700448989868164`

CPU output artifacts:

- Raw: `/workspace/data/outputs/device_cpu/depth_anything_v2_vits/output/Result_0/output.raw`
- Visualization: `/workspace/data/outputs/device_cpu/depth_anything_v2_vits/dechirico_depth_device_cpu.png`

## Backend Decision: DSP or HTP?

QAIRT exposes two related accelerator backend families in this SDK:

- Legacy DSP backend:
  - Host/device backend library: `libQnnDsp.so`
  - Android stub: `libQnnDspV66Stub.so`
  - Hexagon skel: `lib/hexagon-v66/unsigned/libQnnDspV66Skel.so`
- HTP backend:
  - Host/device backend library: `libQnnHtp.so`
  - Android SM8250-relevant stub: `libQnnHtpV68Stub.so`
  - Hexagon skel: `lib/hexagon-v68/unsigned/libQnnHtpV68Skel.so`

The connected phone reports:

```text
ro.product.model = SM-G981N
ro.soc.model     = SM8250
ro.board.platform = kona
ro.hardware      = qcom
ro.product.cpu.abi = arm64-v8a
```

The SDK includes both `hexagon-v66` DSP libraries and `hexagon-v68` HTP libraries. For SM8250/kona, the expected QAIRT path is HTP with `v68` libraries, not the older standalone `DSP v66` path. I will still test the legacy DSP backend as a diagnostic attempt, but the primary target will be `libQnnHtp.so` with `libQnnHtpV68Stub.so` and `libQnnHtpV68Skel.so`.

Important SDK rule from local QAIRT docs:

- HTP and DSP target devices must use quantized models with `--input_list`.
- Android HTP execution requires accelerator-side skel/stub libraries and `ADSP_LIBRARY_PATH`.

## Attempt 1: Platform Validator Setup

Goal:

- Validate whether the device accepts the accelerator backend and which DSP architecture should be used.

Observation:

- Running host-side `qnn-platform-validator --help` after `envsetup.sh` failed due an SDK Python import issue:

```text
ModuleNotFoundError: No module named 'common_utils.adb'
```

Cause:

- The validator wrapper imports `common_utils.adb`, but the QAIRT Python tree under `lib/python/common_utils` contains `protocol/adb.py`. The SDK also ships a separate `benchmarks/QNN/common_utils/adb.py`, so this is an SDK packaging/path issue rather than a model issue.

Next action:

- Use direct Android binary execution and/or backend execution tests instead of relying only on the host wrapper.

Result:

- I pushed the Android validator binary and calculator libraries manually:

```text
/data/local/tmp/qnn_platform_validator_manual/bin/qnn-platform-validator
/data/local/tmp/qnn_platform_validator_manual/lib/libQnnDspV66CalculatorStub.so
/data/local/tmp/qnn_platform_validator_manual/dsp/libCalculator_skel.so
```

- Direct device command:

```bash
cd /data/local/tmp/qnn_platform_validator_manual
export LD_LIBRARY_PATH=/data/local/tmp/qnn_platform_validator_manual/lib
export ADSP_LIBRARY_PATH=/data/local/tmp/qnn_platform_validator_manual/dsp
./bin/qnn-platform-validator \
  --backend dsp \
  --libVersion \
  --coreVersion \
  --testBackend \
  --debug \
  --targetPath /data/local/tmp/qnn_platform_validator_manual/output
```

- Validator output summary:

```text
Backend = DSP
Backend Hardware  : Supported
Backend Libraries : Found
Library Version   : Not Found
Core Version      : Hexagon Architecture V66
Unit Test         : Passed
QNN is supported for backend DSP on the device.
```

Conclusion:

- For this Galaxy S20 SM8250 device, the validated QAIRT accelerator backend is `DSP` with Hexagon `v66`.
- The correct runtime path for the next attempts is therefore:
  - Android backend: `libQnnDsp.so`
  - Android stub: `libQnnDspV66Stub.so`
  - Hexagon skel: `libQnnDspV66Skel.so`
  - Device environment: `LD_LIBRARY_PATH` points to Android `.so` files; `ADSP_LIBRARY_PATH` points to the directory containing Hexagon skel libraries.
- `HTP v68` libraries exist in the SDK, but validator reports this device as DSP v66. HTP will be treated as a secondary diagnostic path, not the primary SM8250 path.

## Attempt 2: Reconvert DLC With Explicit DSP Target and SM8250 SoC

Goal:

- Convert ONNX to a DSP-targeted DLC, using the phone-reported SoC model.

Command:

```bash
source /workspace/qairt/2.45.0.260326/bin/envsetup.sh
qairt-converter \
  --input_network /workspace/models/onnx/depth_anything_v2_vits_518_sim.onnx \
  --source_model_input_shape "input" 1,3,518,518 \
  --source_model_input_layout "input" NCHW \
  --desired_input_layout "input" NCHW \
  --target_backend DSP \
  --target_soc_model SM8250 \
  --output_path /workspace/models/dlc/depth_anything_v2_vits_518_dsp_float.dlc
```

Result:

```text
ERROR - Encountered Error: SOC model SM8250 is not supported.
Exception: SOC model SM8250 is not supported.
```

Interpretation:

- The phone reports `ro.soc.model=SM8250`, but this QAIRT converter build does not have `SM8250` in its SoC-aware converter table.
- This is not a device connection problem and not a model graph problem.

Next action:

- Retry with `--target_backend DSP` only, omitting `--target_soc_model`.

## Attempt 3: Reconvert DLC With Explicit DSP Target Only

Goal:

- Create a DLC while telling the converter that the intended backend is DSP, without SoC-specific optimization.

Command:

```bash
source /workspace/qairt/2.45.0.260326/bin/envsetup.sh
qairt-converter \
  --input_network /workspace/models/onnx/depth_anything_v2_vits_518_sim.onnx \
  --source_model_input_shape "input" 1,3,518,518 \
  --source_model_input_layout "input" NCHW \
  --desired_input_layout "input" NCHW \
  --target_backend DSP \
  --output_path /workspace/models/dlc/depth_anything_v2_vits_518_dsp_float.dlc
```

Result:

```text
addNode: WARNING: Unknown backend name 'DSP' provided. Proceeding without backend-specific handling.
INFO_CONVERSION_SUCCESS: Conversion completed successfully
```

Artifacts:

- `/workspace/models/dlc/depth_anything_v2_vits_518_dsp_float.dlc`

Interpretation:

- The converter accepted the graph and wrote a DLC.
- The warning means this converter path did not apply DSP-specific graph handling. Therefore this is not sufficient proof that the graph can run on DSP; the real validation will happen in quantization and `qnn-net-run`.

Next action:

- Quantize the DLC using the existing float32 calibration input list:
  - `/workspace/data/inputs_raw_fmt/depth_anything_v2_vits/input_list_host.txt`

## Attempt 4: Quantize With `qairt-quantizer --target_backend DSP`

Goal:

- Generate a DSP-targeted quantized DLC.

Command:

```bash
source /workspace/qairt/2.45.0.260326/bin/envsetup.sh
qairt-quantizer \
  --input_dlc /workspace/models/dlc/depth_anything_v2_vits_518_dsp_float.dlc \
  --input_list /workspace/data/inputs_raw_fmt/depth_anything_v2_vits/input_list_host.txt \
  --output_dlc /workspace/models/dlc/depth_anything_v2_vits_518_dsp_quantized.dlc \
  --target_backend DSP \
  --act_bitwidth 8 \
  --weights_bitwidth 8 \
  --bias_bitwidth 32 \
  --use_per_channel_quantization
```

Result:

```text
ERROR - Encountered Error: Backend dsp does not have backend aware config.
ValueError: Backend dsp does not have backend aware config.
```

Interpretation:

- Although `qairt-quantizer --help` lists `DSP` as a supported `--target_backend`, this SDK install does not include a backend-aware quantizer config for `dsp`.
- This blocks the direct `qairt-quantizer --target_backend DSP` path.

Next actions:

1. Try `qairt-quantizer` without `--target_backend`. The tool defaults to HTP, but it may still produce a standard quantized DLC that can be tested with the DSP runtime.
2. Try legacy `snpe-dlc-quantize`, which historically targets DLC quantization for CPU/GPU/DSP SNPE-style runtimes.

## Attempt 5: Quantize With `qairt-quantizer` Default Backend

Goal:

- Produce a quantized DLC despite the missing DSP backend-aware config.

Command:

```bash
source /workspace/qairt/2.45.0.260326/bin/envsetup.sh
qairt-quantizer \
  --input_dlc /workspace/models/dlc/depth_anything_v2_vits_518_dsp_float.dlc \
  --input_list /workspace/data/inputs_raw_fmt/depth_anything_v2_vits/input_list_host.txt \
  --output_dlc /workspace/models/dlc/depth_anything_v2_vits_518_quantized_default.dlc \
  --act_bitwidth 8 \
  --weights_bitwidth 8 \
  --bias_bitwidth 32 \
  --use_per_channel_quantization
```

Result:

```text
WARNING - HTP is provided to backend option but input DLC was generated using backend DSP
Quantization completed successfully
Quantized Model saved at: /workspace/models/dlc/depth_anything_v2_vits_518_quantized_default.dlc
```

Artifact:

- `/workspace/models/dlc/depth_anything_v2_vits_518_quantized_default.dlc`
- Size: about 25 MB

DLC I/O after quantization:

```text
input  : 1,3,518,518 uFxp_8
output : 1,518,518   uFxp_8
```

Note:

- `qnn-net-run` can still consume the existing float32 `.raw` if `--use_native_input_files` is not passed; it quantizes the input internally according to DLC encodings. Likewise, without `--use_native_output_files`, it writes float output files for convenient comparison.

## Attempt 6: Quantize With Legacy `snpe-dlc-quantize`

Goal:

- Generate a second quantized DLC through the legacy SNPE quantizer, because SM8250 exposes a legacy DSP v66 path.

Command:

```bash
source /workspace/qairt/2.45.0.260326/bin/envsetup.sh
snpe-dlc-quantize \
  --input_dlc /workspace/models/dlc/depth_anything_v2_vits_518_float.dlc \
  --input_list /workspace/data/inputs_raw_fmt/depth_anything_v2_vits/input_list_host.txt \
  --output_dlc /workspace/models/dlc/depth_anything_v2_vits_518_snpe_quantized.dlc \
  --use_enhanced_quantizer
```

Result:

```text
Successfully saved DLC to /workspace/models/dlc/depth_anything_v2_vits_518_snpe_quantized.dlc
```

Artifact:

- `/workspace/models/dlc/depth_anything_v2_vits_518_snpe_quantized.dlc`
- Size: about 24 MB

Note:

- The `--use_enhanced_quantizer` flag is deprecated, but accepted by this SDK.
- This DLC will be tested after the default `qairt-quantizer` DLC if needed.

## Attempt 7: Host CPU Sanity Check for Quantized DLCs

Goal:

- Make sure the quantized DLCs can at least compose/finalize/execute through `qnn-net-run` before trying the phone DSP backend.

Commands:

```bash
qnn-net-run \
  --backend /workspace/qairt/2.45.0.260326/lib/x86_64-linux-clang/libQnnCpu.so \
  --model /workspace/qairt/2.45.0.260326/lib/x86_64-linux-clang/libQnnModelDlc.so \
  --dlc_path /workspace/models/dlc/depth_anything_v2_vits_518_quantized_default.dlc \
  --input_list /workspace/data/inputs_raw_fmt/depth_anything_v2_vits/input_list_host.txt \
  --output_dir /workspace/data/outputs/qnn_host_cpu/depth_anything_v2_vits_quant_default

qnn-net-run \
  --backend /workspace/qairt/2.45.0.260326/lib/x86_64-linux-clang/libQnnCpu.so \
  --model /workspace/qairt/2.45.0.260326/lib/x86_64-linux-clang/libQnnModelDlc.so \
  --dlc_path /workspace/models/dlc/depth_anything_v2_vits_518_snpe_quantized.dlc \
  --input_list /workspace/data/inputs_raw_fmt/depth_anything_v2_vits/input_list_host.txt \
  --output_dir /workspace/data/outputs/qnn_host_cpu/depth_anything_v2_vits_snpe_quant
```

Result:

- Both quantized DLCs executed on host CPU.

Accuracy against ONNX float baseline:

| DLC | MSE | MAE | MAX_ABS | Output Min | Output Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| `depth_anything_v2_vits_518_quantized_default.dlc` | `5.329611` | `1.721263` | `5.931912` | `0.309530` | `0.619060` |
| `depth_anything_v2_vits_518_snpe_quantized.dlc` | `5.162710` | `1.709319` | `5.842595` | `0.383022` | `0.718167` |

Interpretation:

- The DLCs are executable, but one-image post-training quantization is very inaccurate for this transformer-heavy model.
- This is not yet the final quality target. The immediate next goal is accelerator execution. Accuracy improvements will likely require better calibration data, graph edits, mixed precision, or model-level changes.

## Attempt 8: Run Quantized DLCs on Galaxy S20 DSP

Goal:

- Run both quantized DLC candidates on the validated DSP v66 backend.

Device setup:

```bash
TARGET=/data/local/tmp/depth_anything_v2_vits_dsp
SDK=/workspace/qairt/2.45.0.260326

adb shell "rm -rf $TARGET && mkdir -p $TARGET/input $TARGET/output $TARGET/dsp"
adb push $SDK/bin/aarch64-android/qnn-net-run $TARGET/
adb push $SDK/lib/aarch64-android/libQnnDsp.so $TARGET/
adb push $SDK/lib/aarch64-android/libQnnDspV66Stub.so $TARGET/
adb push $SDK/lib/aarch64-android/libQnnModelDlc.so $TARGET/
adb push $SDK/lib/aarch64-android/libQnnSystem.so $TARGET/
adb push $SDK/lib/hexagon-v66/unsigned/libQnnDspV66Skel.so $TARGET/dsp/
adb push $SDK/lib/hexagon-v66/unsigned/libQnnDspV66.so $TARGET/dsp/
adb push $SDK/lib/hexagon-v66/unsigned/libQnnSystem.so $TARGET/dsp/
```

Runtime environment:

```bash
export LD_LIBRARY_PATH=/data/local/tmp/depth_anything_v2_vits_dsp
export ADSP_LIBRARY_PATH="/data/local/tmp/depth_anything_v2_vits_dsp/dsp;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp"
```

Command shape:

```bash
./qnn-net-run \
  --backend libQnnDsp.so \
  --model libQnnModelDlc.so \
  --dlc_path <quantized.dlc> \
  --input_list input_list_target.txt \
  --output_dir <output_dir> \
  --log_level verbose
```

Result for both quantized DLCs:

```text
Failed in QnnSystemDlc_composeGraphs() with error code 1002
Received nullptr for graphsInput.
Failed to copy graph infos while composing graphs from dlc.
ComposeGraphs Failed with error = 1
Graph Prepare failure
```

The useful detail appeared in `adb logcat`:

```text
NATIVE OpValidator::validateOpConfig /model/blocks.0/attn/Gather:qti.aisw:Gather
Input[0] has incorrect Rank 5.
Exception encountered: Validate OpConfig failed:
QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE
```

Root cause:

- The original DINOv2 attention code creates `qkv` as a rank-5 tensor and indexes it:

```python
qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
```

- ONNX export represents `qkv[0]`, `qkv[1]`, and `qkv[2]` as `Gather` operations on a rank-5 tensor.
- The SM8250 DSP v66 backend rejects this rank-5 `Gather`.

Required model/export change:

- Rewrite attention to split the linear output while it is still rank-3, before reshaping into heads.
- Desired structure:

```python
qkv = self.qkv(x)
q, k, v = qkv.chunk(3, dim=-1)
q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
k = k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
v = v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
```

Expected effect:

- Remove the rank-5 `Gather` pattern from attention.
- Replace it with rank-3 `Split`/slice plus reshape/transpose, which is more likely to pass DSP validation.

## Attempt 9: Patch Attention and Re-export ONNX

Code change:

- File: `/workspace/external/Depth-Anything-V2/depth_anything_v2/dinov2_layers/attention.py`
- Original code indexed Q/K/V from a rank-5 tensor.
- Patched code chunks the final linear output while it is still rank-3, then reshapes each of Q/K/V separately.

Patch essence:

```python
head_dim = C // self.num_heads
q, k, v = self.qkv(x).chunk(3, dim=-1)
q = q.reshape(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)
k = k.reshape(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)
v = v.reshape(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)
q = q * self.scale
```

New export config:

- `/workspace/configs/depth_model_dsp.yaml`

New ONNX:

- `/workspace/models/onnx/depth_anything_v2_vits_518_dsp_gatherfix.onnx`

Validation:

- `torch.onnx.export` completed.
- ONNX structural validation passed.
- PyTorch vs ONNX Runtime validation passed.
- Shape-inferred ONNX graph check:

```text
Gather count: 12
Rank>=5 Gather count: 0
```

Interpretation:

- The graph no longer contains the exact rank-5 `Gather` pattern that failed on DSP v66.

## Attempt 10: Convert, Quantize, and Run Gather-Fixed Graph on DSP

Commands:

```bash
qairt-converter \
  --input_network /workspace/models/onnx/depth_anything_v2_vits_518_dsp_gatherfix.onnx \
  --source_model_input_shape "input" 1,3,518,518 \
  --source_model_input_layout "input" NCHW \
  --desired_input_layout "input" NCHW \
  --target_backend DSP \
  --output_path /workspace/models/dlc/depth_anything_v2_vits_518_dsp_gatherfix_float.dlc

qairt-quantizer \
  --input_dlc /workspace/models/dlc/depth_anything_v2_vits_518_dsp_gatherfix_float.dlc \
  --input_list /workspace/data/inputs_raw_fmt/depth_anything_v2_vits/input_list_host.txt \
  --output_dlc /workspace/models/dlc/depth_anything_v2_vits_518_dsp_gatherfix_quantized.dlc \
  --act_bitwidth 8 \
  --weights_bitwidth 8 \
  --bias_bitwidth 32 \
  --use_per_channel_quantization
```

Result:

- Conversion succeeded.
- Quantization succeeded.
- Host CPU sanity run succeeded.
- DLC op list no longer contains `Gather`.

DSP run result:

```text
Failed in QnnSystemDlc_composeGraphs() with error code 1002
ComposeGraphs Failed with error = 1
Graph Prepare failure
```

Useful `logcat` detail:

```text
NATIVE OpValidator::validateOpConfig _layernorm_0:qti.aisw:LayerNorm
Input[2] has incorrect Datatype 0x332.
Exception encountered: Validate OpConfig failed:
QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE
```

DLC info around the first LayerNorm:

```text
_layernorm_0 LayerNorm
input  : /model/Add_output_0, uFxp_8, [1,1370,384]
weight : model.pretrained.blocks.0.norm1.weight, uFxp_8, [384]
bias   : model.pretrained.blocks.0.norm1.bias, sFxp_32, [384]
axes   : [2]
```

Interpretation:

- The rank-5 Gather issue is fixed.
- The next blocker is DSP v66 LayerNorm validation.
- The quantizer converted LayerNorm bias tensors to `sFxp_32`, but DSP v66 LayerNorm rejects that parameter datatype.

Next action:

- Try preserving LayerNorm-related parameters in float, or rewrite LayerNorm into primitive arithmetic if preservation is not accepted.

## Attempt 11: Quantize Gather-Fixed Graph With 8-bit Bias

Goal:

- Avoid `sFxp_32` LayerNorm bias tensors, because DSP v66 rejected `LayerNorm` input 2 when it was `sFxp_32`.

Command:

```bash
qairt-quantizer \
  --input_dlc /workspace/models/dlc/depth_anything_v2_vits_518_dsp_gatherfix_float.dlc \
  --input_list /workspace/data/inputs_raw_fmt/depth_anything_v2_vits/input_list_host.txt \
  --output_dlc /workspace/models/dlc/depth_anything_v2_vits_518_dsp_gatherfix_quantized_bias8.dlc \
  --act_bitwidth 8 \
  --weights_bitwidth 8 \
  --bias_bitwidth 8 \
  --use_per_channel_quantization
```

Result:

- Quantization succeeded.
- Host CPU sanity run succeeded.
- First LayerNorm now has 8-bit weight and 8-bit bias:

```text
model.pretrained.blocks.0.norm1.weight: uFxp_8
model.pretrained.blocks.0.norm1.bias  : uFxp_8
```

DSP run result:

```text
Graph Prepare failure
```

Useful `logcat` detail:

```text
NATIVE OpValidator::validateOpConfig _elementwiseneuron_0:qti.aisw:ElementWiseNeuron
Param[0] has incorrect Value 1.
Exception encountered: Validate OpConfig failed:
QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE
```

Interpretation:

- The LayerNorm datatype blocker was fixed by using `--bias_bitwidth 8`.
- The next blocker is the transformer MLP activation. In the original DINOv2 ViT blocks, the MLP uses `nn.GELU`; QAIRT converted this to `ElementWiseNeuron` operation value `1`, which DSP v66 rejects.

Next action:

- Replace DINOv2 MLP activation from `GELU` to `ReLU` for a DSP smoke-test graph.
- This is a real model behavior change and will reduce accuracy, but it tells us whether the rest of the graph can pass DSP composition/execution.

## Attempt 12: ReLU Activation Smoke Test

Goal:

- Prove that the remaining graph can actually run on SM8250 DSP once the unsupported GELU form is removed.
- This attempt intentionally changes model behavior. It is a DSP bring-up test, not the final accuracy-preserving model.

Code/config changes:

- Added a configurable DINOv2 MLP activation path:
  - `/workspace/external/Depth-Anything-V2/depth_anything_v2/dinov2.py`
  - `/workspace/external/Depth-Anything-V2/depth_anything_v2/dpt.py`
  - `/workspace/src/models/depth_anything_v2_vits.py`
- Default activation is still `gelu`, matching the original model.
- The ReLU experiment is selected only by config:
  - `/workspace/configs/depth_model_dsp_relu.yaml`
  - `model.params.dinov2_act_layer: relu`

Export/convert/quantize artifacts:

```text
/workspace/models/onnx/depth_anything_v2_vits_518_dsp_gatherfix_relu.onnx
/workspace/models/dlc/depth_anything_v2_vits_518_dsp_gatherfix_relu_float.dlc
/workspace/models/dlc/depth_anything_v2_vits_518_dsp_gatherfix_relu_quantized_bias8.dlc
```

Important quantization setting:

```bash
--act_bitwidth 8 \
--weights_bitwidth 8 \
--bias_bitwidth 8 \
--use_per_channel_quantization
```

Reason:

- `--bias_bitwidth 32` caused DSP LayerNorm validation failure.
- `--bias_bitwidth 8` keeps LayerNorm weight/bias as 8-bit tensors and passes that blocker.

Device run command:

```bash
adb shell "cd /data/local/tmp/depth_anything_v2_vits_dsp && \
  rm -rf output_relu_bias8 && mkdir -p output_relu_bias8 && \
  export LD_LIBRARY_PATH=/data/local/tmp/depth_anything_v2_vits_dsp && \
  export ADSP_LIBRARY_PATH='/data/local/tmp/depth_anything_v2_vits_dsp/dsp;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp' && \
  ./qnn-net-run \
    --backend libQnnDsp.so \
    --model libQnnModelDlc.so \
    --dlc_path depth_anything_v2_vits_518_dsp_gatherfix_relu_quantized_bias8.dlc \
    --input_list input_list_target_dsp.txt \
    --output_dir output_relu_bias8 \
    --log_level verbose"
```

Result:

- DSP compose succeeded.
- DSP finalize succeeded.
- DSP execute succeeded.
- Output raw was generated:

```text
/data/local/tmp/depth_anything_v2_vits_dsp/output_relu_bias8/Result_0/output.raw
/workspace/data/outputs/device_dsp/depth_anything_v2_vits_relu_bias8/output_relu_bias8/Result_0/output.raw
```

Execution metadata says the graph used quantized I/O internally:

```text
input : QNN_DATATYPE_UFIXED_POINT_8, [1,3,518,518]
output: QNN_DATATYPE_UFIXED_POINT_8, [1,518,518]
```

Because `qnn-net-run` was not called with `--use_native_output_files`, the saved `output.raw` is dequantized float32.

Host CPU vs device DSP for the same ReLU DLC:

```text
MSE     : 0.0001273923
MAE     : 0.00825307
MAX_ABS : 0.0442723
Corr    : 0.991826
```

Original CPU model vs device DSP ReLU:

```text
MSE     : 3.275555
MAE     : 1.474604
MAX_ABS : 4.474786
Corr    : 0.858297
```

Generated visualization:

```text
/workspace/data/outputs/device_dsp/depth_anything_v2_vits_relu_bias8/dechirico_depth_device_dsp_relu_bias8.png
/workspace/data/outputs/device_dsp/depth_anything_v2_vits_relu_bias8/compare_device_cpu_original_vs_device_dsp_relu_bias8.png
```

Interpretation:

- This is the first successful real SM8250 DSP inference for the model family.
- However, replacing GELU with ReLU compresses the depth structure heavily. It is useful as proof that attention, LayerNorm, Softmax, Resize, Conv, and MatMul can run on DSP after the previous fixes, but it is not the best quality candidate.

## Attempt 13: ONNX Opset 20 GELU Export

Goal:

- Keep the original GELU activation by exporting ONNX with opset 20, where `Gelu` can appear as an ONNX op instead of the opset-13 `Erf` decomposition.
- Check whether QAIRT preserves it as a backend-supported `Gelu` op instead of converting it to unsupported `ElementWiseNeuron operation: 1`.

Config:

```text
/workspace/configs/depth_model_dsp_opset20.yaml
```

Exported ONNX:

```text
/workspace/models/onnx/depth_anything_v2_vits_518_dsp_gatherfix_opset20.onnx
```

ONNX op count confirmed:

```text
Gelu               12
LayerNormalization 28
Gather             12
```

Converted DLC:

```text
/workspace/models/dlc/depth_anything_v2_vits_518_dsp_gatherfix_opset20_float.dlc
```

DLC inspection result:

```text
/model/blocks.0/mlp/act/Gelu  ElementWiseNeuron  operation: 1
...
```

Interpretation:

- Opset 20 changes the ONNX graph, but QAIRT still lowers GELU to `ElementWiseNeuron operation: 1`.
- This is the same form rejected by the SM8250 DSP v66 validator.
- Therefore opset 20 alone does not solve the original GELU blocker.

## Attempt 14: SiLU Activation Approximation

Goal:

- Try an activation closer to GELU than ReLU while staying in DSP-supported primitive territory.
- `SiLU(x) = x * sigmoid(x)` exports as `Sigmoid` + `Mul`, not as GELU.

Code/config:

- Added `silu` to the activation resolver in:
  - `/workspace/external/Depth-Anything-V2/depth_anything_v2/dinov2.py`
- Config:
  - `/workspace/configs/depth_model_dsp_silu.yaml`

Artifacts:

```text
/workspace/models/onnx/depth_anything_v2_vits_518_dsp_gatherfix_silu.onnx
/workspace/models/dlc/depth_anything_v2_vits_518_dsp_gatherfix_silu_float.dlc
/workspace/models/dlc/depth_anything_v2_vits_518_dsp_gatherfix_silu_quantized_bias8.dlc
```

DLC inspection:

```text
/model/blocks.0/mlp/act/Sigmoid  ElementWiseNeuron  operation: 6
/model/blocks.0/mlp/act/Mul      Eltwise_Binary     operation: 13
```

DSP result:

- Compose succeeded.
- Finalize succeeded.
- Execute succeeded.

Output:

```text
/workspace/data/outputs/device_dsp/depth_anything_v2_vits_silu_bias8/output_silu_bias8/Result_0/output.raw
/workspace/data/outputs/device_dsp/depth_anything_v2_vits_silu_bias8/dechirico_depth_device_dsp_silu_bias8.png
```

Metrics:

```text
Host CPU SiLU DLC vs device DSP SiLU:
MSE     : 0.00924418
MAE     : 0.0280021
MAX_ABS : 1.27283
Corr    : 0.862745

Original CPU model vs device DSP SiLU:
MSE     : 6.473019
MAE     : 1.761315
MAX_ABS : 6.284513
Corr    : 0.269111
```

Interpretation:

- DSP accepts `Sigmoid operation: 6`.
- Plain SiLU runs on DSP but produces a poor depth map for this checkpoint. It is worse than the ReLU smoke-test in correlation with the original CPU output.

## Attempt 15: QuickGELU Approximation

Goal:

- Use a GELU-like approximation that avoids DSP-rejected `ElementWiseNeuron operation: 1`.
- `QuickGELU(x) = x * sigmoid(1.702 * x)` is much closer to GELU than plain SiLU and still exports as `Sigmoid` + `Mul`.

Code/config:

- Added `QuickGELU` to:
  - `/workspace/external/Depth-Anything-V2/depth_anything_v2/dinov2.py`

```python
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)
```

- Config:
  - `/workspace/configs/depth_model_dsp_quick_gelu.yaml`
  - `model.params.dinov2_act_layer: quick_gelu`

Artifacts:

```text
/workspace/models/onnx/depth_anything_v2_vits_518_dsp_gatherfix_quick_gelu.onnx
/workspace/models/dlc/depth_anything_v2_vits_518_dsp_gatherfix_quick_gelu_float.dlc
/workspace/models/dlc/depth_anything_v2_vits_518_dsp_gatherfix_quick_gelu_quantized_bias8.dlc
```

Device run command:

```bash
adb shell "cd /data/local/tmp/depth_anything_v2_vits_dsp && \
  rm -rf output_quick_gelu_bias8 && mkdir -p output_quick_gelu_bias8 && \
  export LD_LIBRARY_PATH=/data/local/tmp/depth_anything_v2_vits_dsp && \
  export ADSP_LIBRARY_PATH='/data/local/tmp/depth_anything_v2_vits_dsp/dsp;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp' && \
  ./qnn-net-run \
    --backend libQnnDsp.so \
    --model libQnnModelDlc.so \
    --dlc_path depth_anything_v2_vits_518_dsp_gatherfix_quick_gelu_quantized_bias8.dlc \
    --input_list input_list_target_dsp.txt \
    --output_dir output_quick_gelu_bias8 \
    --log_level verbose"
```

DSP result:

- Compose succeeded.
- Finalize succeeded.
- Execute succeeded.

Output:

```text
/workspace/data/outputs/device_dsp/depth_anything_v2_vits_quick_gelu_bias8/output_quick_gelu_bias8/Result_0/output.raw
/workspace/data/outputs/device_dsp/depth_anything_v2_vits_quick_gelu_bias8/dechirico_depth_device_dsp_quick_gelu_bias8.png
/workspace/data/outputs/device_dsp/depth_anything_v2_vits_quick_gelu_bias8/dechirico_depth_device_dsp_quick_gelu_bias8_gray.png
/workspace/data/outputs/device_dsp/depth_anything_v2_vits_quick_gelu_bias8/compare_cpu_original_relu_silu_quick_gelu.png
```

Metrics:

```text
Host CPU QuickGELU DLC vs device DSP QuickGELU:
MSE     : 0.00793274
MAE     : 0.0747298
MAX_ABS : 1.409127
Corr    : 0.997139

Original CPU model vs device DSP QuickGELU:
MSE     : 1.652901
MAE     : 1.049206
MAX_ABS : 3.870214
Corr    : 0.958544
```

Comparison against other DSP-successful variants:

```text
Original CPU vs DSP ReLU:
MAE 1.474604, Corr 0.858297

Original CPU vs DSP SiLU:
MAE 1.761315, Corr 0.269111

Original CPU vs DSP QuickGELU:
MAE 1.049206, Corr 0.958544
```

Interpretation:

- QuickGELU is the best current DSP-compatible activation replacement.
- It keeps the graph away from the unsupported GELU `operation: 1`, but preserves much more of the original depth structure than ReLU or SiLU.
- This is still not the pure original Depth Anything V2 model. It is a controlled DSP-compatible approximation:
  - attention Q/K/V export rewritten,
  - LayerNorm bias quantized to 8-bit,
  - DINOv2 MLP activation changed from GELU to QuickGELU.

## Current Best Reproduction Path

Use this path when reproducing the current best SM8250 DSP result:

1. Keep the attention patch in:

```text
/workspace/external/Depth-Anything-V2/depth_anything_v2/dinov2_layers/attention.py
```

2. Use QuickGELU config:

```text
/workspace/configs/depth_model_dsp_quick_gelu.yaml
```

3. Export ONNX:

```bash
PYTHONPATH=/workspace python3 /workspace/src/export_onnx.py \
  --config /workspace/configs/depth_model_dsp_quick_gelu.yaml
```

4. Convert to DLC:

```bash
source /workspace/qairt/2.45.0.260326/bin/envsetup.sh
qairt-converter \
  --input_network /workspace/models/onnx/depth_anything_v2_vits_518_dsp_gatherfix_quick_gelu.onnx \
  --source_model_input_shape input 1,3,518,518 \
  --target_backend DSP \
  --output_path /workspace/models/dlc/depth_anything_v2_vits_518_dsp_gatherfix_quick_gelu_float.dlc
```

5. Quantize:

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

6. Push and run on device:

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

Backend conclusion remains:

- Galaxy S20 SM8250 reports Hexagon Architecture `V66` through `qnn-platform-validator`.
- Use the QNN DSP path (`libQnnDsp.so`, `libQnnDspV66Stub.so`, `libQnnDspV66Skel.so`).
- Do not use the HTP v68 path for this device. In this QAIRT package the quantizer/converter may warn about HTP defaults, but the actual device runtime path used here is DSP v66.

## Attempt 16: Re-check Runtime Measurement Methodology

Reason:

- The first README runtime table reported the QuickGELU DSP path at about 106 seconds.
- This is counter-intuitive because DSP acceleration is usually expected to be faster than CPU execution for supported neural-network workloads.
- Re-measurement was needed to check whether the number was a timing artifact.

Measurement method:

- Used the same host-side Python `time.perf_counter()` wrapper for all three targets.
- Used the same final QuickGELU quantized DLC:

```text
/workspace/models/dlc/depth_anything_v2_vits_518_dsp_gatherfix_quick_gelu_quantized_bias8.dlc
```

- Each run created a valid output tensor of `1,073,296` bytes.
- Measurements are end-to-end `qnn-net-run` wall-clock times, not pure kernel timings.

Cold single-run result:

```text
host_cpu_cold    run 0:   0.762 s
host_cpu_cold    run 1:   0.742 s
device_cpu_cold  run 0:   1.667 s
device_cpu_cold  run 1:   1.683 s
device_dsp_cold  run 0: 106.842 s
device_dsp_cold  run 1: 106.826 s
```

Conclusion from cold runs:

- The 106-second DSP timing is reproducible.
- It is not caused by Android `date` clock drift or a one-off measurement error.

Additional `--num_inferences 2 --keep_num_outputs 1` result:

```text
host_cpu     total:   1.451 s, average:   0.725 s / inference
device_cpu   total:   3.134 s, average:   1.567 s / inference
device_dsp   total: 212.709 s, average: 106.354 s / inference
```

Conclusion from two-inference runs:

- The DSP delay is not primarily graph initialization or one-time graph finalization.
- The execution path remains around 106 seconds per inference even after amortization.
- Therefore the current DSP path is a correctness/feasibility milestone only.

Interpretation:

- SM8250 DSP v66 successfully executes the final quantized graph, but this full-resolution transformer workload is not currently optimized for latency on that backend.
- The graph contains large attention tensors, MatMul, Softmax, LayerNorm, Slice/Reshape/Transpose, and other transformer-style operations rather than a DSP-friendly CNN workload.
- Future performance work should profile per-op cost, test cached context binaries, reduce resolution, broaden calibration, and consider a mobile-distilled architecture.
