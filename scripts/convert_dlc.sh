# onnx -> dlc 변환 (qairt-converter 활용)
#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source "${ROOT_DIR}/configs/env_setup.sh"

INPUT_ONNX="${ROOT_DIR}/models/onnx/depth_anything_v2_vits_518_sim.onnx"
OUTPUT_DLC="${ROOT_DIR}/models/dlc/depth_anything_v2_vits_518_sim.dlc"

mkdir -p "${ROOT_DIR}/models/dlc"

qairt-converter \
  --input_network "${INPUT_ONNX}" \
  --source_model_input_shape "input" 1,3,518,518 \
  --output_path "${OUTPUT_DLC}"

echo "DLC conversion complete: ${OUTPUT_DLC}"