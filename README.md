# QAIRT Model Porting for Lightweight Monocular Depth Estimation

This repository documents an end-to-end deployment study for porting lightweight monocular depth estimation models to a Samsung Galaxy S20 using Qualcomm AI Runtime (QAIRT). The project is being developed as an undergraduate research and engineering effort in the Efficient & Versatile AI Lab (EV Lab), Konkuk University.

The current milestone is not the final depth model deployment itself. Instead, we have already verified the complete deployment pipeline with a minimal `SimpleCNN` smoke-test model:

- PyTorch model definition and ONNX export
- ONNX Runtime validation on the host
- ONNX to DLC and quantized DLC conversion for QAIRT
- ADB-based model and input transfer to Galaxy S20
- On-device inference on CPU and DSP backends
- Output collection and numerical consistency analysis across host, CPU, and DSP results

This verified pipeline now serves as the foundation for the next phase: selecting a lightweight monocular depth estimation network and porting it to the target mobile device under QAIRT constraints.

## Project Objectives

- Establish a reproducible QAIRT deployment workflow from model export to on-device validation.
- Build an engineering baseline for mobile inference on Galaxy S20 before integrating a real depth model.
- Prepare the codebase for graph surgery, unsupported operator handling, quantization, and backend-specific optimization.
- Transition from a controlled smoke test to a practical lightweight monocular depth estimation model suitable for mobile deployment.

## Current Status

The repository currently contains validated artifacts from the `SimpleCNN` pipeline. This control experiment confirms that the software environment, device communication, runtime invocation flow, and output comparison logic are functioning as intended.

Observed output agreement from the existing validation artifacts:

| Comparison | MSE | MAE | Max Abs |
| --- | ---: | ---: | ---: |
| Host vs CPU | 1.10e-15 | 2.58e-08 | 1.34e-07 |
| Host vs DSP | 5.16e-06 | 2.08e-03 | 5.14e-03 |
| CPU vs DSP | 5.16e-06 | 2.08e-03 | 5.14e-03 |

These numbers indicate near-identical host and CPU outputs, while the DSP path shows a small quantization-induced deviation that remains well behaved for a smoke-test deployment.

## Repository Structure

```text
qairt-model-porting/
├── configs/
│   ├── depth_model.yaml         # Placeholder for the target depth model configuration
│   ├── env_setup.sh             # Placeholder for QAIRT SDK environment setup
│   └── simple_cnn.yaml          # Smoke-test model configuration for export and validation
├── data/
│   ├── raw/                     # Raw source images or evaluation inputs
│   ├── inputs_raw_fmt/          # QAIRT-ready .raw inputs and calibration/input lists
│   └── outputs/                 # Host, CPU, and DSP inference outputs
├── docs/
│   └── op_workarounds.md        # Notes on unsupported operator workarounds and graph edits
├── models/
│   ├── pytorch/                 # Source checkpoints or original PyTorch weights
│   ├── onnx/                    # Exported ONNX models
│   └── dlc/                     # QAIRT-ready DLC and quantized DLC artifacts
├── qairt/
│   └── 2.44.0.260225/           # Local QAIRT SDK distribution (not intended for Git tracking)
├── scripts/
│   ├── convert_dlc.sh           # Entry point for ONNX -> DLC conversion automation
│   ├── device_run.sh            # Entry point for ADB push/run/pull automation
│   └── export_onnx.sh           # Wrapper for PyTorch -> ONNX export
├── src/
│   ├── data_utils/              # Reserved for input/output preprocessing utilities
│   ├── graph_surgery/           # Reserved for ONNX graph modifications and operator workarounds
│   ├── models/
│   │   └── simple_cnn.py        # Minimal CNN used for pipeline verification
│   ├── evaluate.py              # Numerical comparison of host/device outputs
│   ├── export_onnx.py           # ONNX export and ONNX Runtime consistency validation
│   └── server_infer.py          # Host-side ONNX Runtime baseline inference
├── tests/
│   └── test_mse.py              # Placeholder for tolerance-based regression tests
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

## System Environment

- Target device: Samsung Galaxy S20
- Target runtime: Qualcomm AI Runtime (QAIRT / QNN)
- Current validated backend path: CPU and DSP
- Container base image: Ubuntu 22.04
- Core software stack: Python, PyTorch, ONNX, ONNX Runtime, ADB

The repository is designed around a Docker-based workflow so that export, validation, and device interaction happen in a reproducible environment.

## Docker-Based Setup

Build the development image from the project root:

```bash
docker build -t qairt-env:latest .
```

Run the container with host networking and host ADB access:

```bash
docker run -it --name qairt-dev \
  --network host \
  -e ADB_SERVER_SOCKET=tcp:127.0.0.1:5037 \
  -v $PWD:/workspace \
  qairt-env:latest bash
```

### Notes

- The repository is mounted to `/workspace` inside the container.
- Current Python scripts use `/workspace/...` paths, so they should be executed inside the container.
- The local `qairt/` directory is treated as an external SDK payload and is intentionally excluded from version control.
- ADB should already be available on the host, and the target Galaxy S20 should be connected with USB debugging enabled.

## Validated Pipeline

The currently verified deployment flow is:

1. Define a minimal PyTorch model in `src/models/simple_cnn.py`.
2. Export the model to ONNX with `src/export_onnx.py` and validate the exported graph using ONNX Runtime.
3. Convert the ONNX model into QAIRT DLC artifacts, including a quantized DLC for DSP execution.
4. Push the model, backend libraries, and `.raw` inputs to the Galaxy S20 through ADB.
5. Run `qnn-net-run` on CPU and DSP backends.
6. Pull output tensors back to the host machine.
7. Compare host, CPU, and DSP outputs with MSE, MAE, and maximum absolute error.

Representative source files for this workflow:

- `scripts/export_onnx.sh`
- `src/export_onnx.py`
- `src/server_infer.py`
- `src/evaluate.py`

At the moment, `scripts/convert_dlc.sh` and `scripts/device_run.sh` act as shell entry points for the next stage of full automation. The validated artifacts already present in `models/` and `data/outputs/` confirm that the end-to-end flow has been exercised successfully.

## Quick Start Inside the Container

Export the smoke-test model to ONNX:

```bash
./scripts/export_onnx.sh
```

Run host-side baseline inference:

```bash
python3 src/server_infer.py
```

Evaluate numerical differences between host and device outputs:

```bash
python3 src/evaluate.py
```

## Engineering Significance

This project is fundamentally about more than exporting a single model. It is an exercise in system-level deployment engineering for edge AI:

- aligning model interfaces across PyTorch, ONNX, and QAIRT;
- verifying numerical stability across heterogeneous execution backends;
- preparing for unsupported operator replacement and graph surgery;
- and building a deployment methodology that can be reused for real mobile depth estimation models.

By validating the infrastructure first with a controlled CNN baseline, the project reduces integration risk before moving to a more complex monocular depth estimation network.

## Next Steps

- Select a lightweight monocular depth estimation architecture appropriate for mobile deployment.
- Implement input preprocessing and output postprocessing tailored to the chosen depth model.
- Expand `src/graph_surgery/` to handle unsupported operators or QAIRT conversion constraints.
- Automate DLC conversion and on-device execution scripts more completely.
- Benchmark latency, numerical error, and deployment feasibility on Galaxy S20.

## Affiliation

Efficient & Versatile AI Lab (EV Lab)  
Konkuk University
