#!/usr/bin/env bash
#
# This script performs both DLC conversion and quantization for Depth Anything V2 (ViT-S)
# specifically optimized for the Galaxy S20 (SM8250) DSP backend.
#
# Usage:
#   ./scripts/prepare_dsp_dlc.sh
#

set -e

# 1. Path Configuration
# Use the workspace root for local or absolute paths
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# QAIRT SDK path
# Default to the version in README, but fallback to whatever is in the qairt/ directory
QAIRT_SDK="/workspace/qairt/2.45.0.260326"
if [ ! -d "${QAIRT_SDK}" ]; then
    # Fallback to local project directory structure
    LOCAL_QAIRT_DIR="${ROOT_DIR}/qairt"
    if [ -d "${LOCAL_QAIRT_DIR}" ]; then
        # Pick the first directory inside qairt/
        DETECTED_SDK=$(ls -d "${LOCAL_QAIRT_DIR}"/* 2>/dev/null | head -n 1)
        if [ -n "${DETECTED_SDK}" ]; then
            QAIRT_SDK="${DETECTED_SDK}"
        fi
    fi
fi

# Model and Data Paths
INPUT_RES="252"

INPUT_ONNX="${ROOT_DIR}/models/onnx/depth_anything_v2_vits_${INPUT_RES}_base.onnx"
FLOAT_DLC="${ROOT_DIR}/models/dlc/depth_anything_v2_vits_${INPUT_RES}_base_float.dlc"
QUANT_DLC="${ROOT_DIR}/models/dlc/depth_anything_v2_vits_${INPUT_RES}_base_quantized_bias8.dlc"
INPUT_LIST="${ROOT_DIR}/data/inputs_raw_fmt/depth_anything_v2_vits/input_list_host.txt"

# 2. Environment Setup
if [ -f "${QAIRT_SDK}/bin/envsetup.sh" ]; then
    echo "[INFO] Sourcing QAIRT environment from ${QAIRT_SDK}"
    source "${QAIRT_SDK}/bin/envsetup.sh"
else
    echo "[WARNING] QAIRT SDK not found at ${QAIRT_SDK}. Attempting to use existing environment."
fi

# Ensure output directory exists
mkdir -p "$(dirname "${FLOAT_DLC}")"

# Resolution setup (Matching your YAML config)

echo ""
echo "========================================================"
echo " STEP 1: DLC Conversion (Float)"
echo "========================================================"
echo "Input  : ${INPUT_ONNX}"
echo "Output : ${FLOAT_DLC}"
echo "Shape  : 1,3,${INPUT_RES},${INPUT_RES}"
echo "Target : DSP"
echo "--------------------------------------------------------"

qairt-converter \
  --input_network "${INPUT_ONNX}" \
  --source_model_input_shape "input" 1,3,${INPUT_RES},${INPUT_RES} \
  --source_model_input_layout "input" NCHW \
  --desired_input_layout "input" NCHW \
  --target_backend DSP \
  --output_path "${FLOAT_DLC}"

echo ""
echo "========================================================"
echo " STEP 2: DLC Quantization (INT8 + Bias8)"
echo "========================================================"
echo "Input  : ${FLOAT_DLC}"
echo "Output : ${QUANT_DLC}"
echo "Config : 8-bit act/weight/bias, per-channel"
echo "--------------------------------------------------------"

qairt-quantizer \
  --input_dlc "${FLOAT_DLC}" \
  --input_list "${INPUT_LIST}" \
  --output_dlc "${QUANT_DLC}" \
  --act_bitwidth 8 \
  --weights_bitwidth 8 \
  --bias_bitwidth 8 \
  --use_per_channel_quantization

echo ""
echo "========================================================"
echo " SUCCESS!"
echo "--------------------------------------------------------"
echo "Final Artifact: ${QUANT_DLC}"
echo "========================================================"
