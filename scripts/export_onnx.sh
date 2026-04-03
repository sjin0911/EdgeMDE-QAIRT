#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-$PROJECT_ROOT/configs/simple_cnn.yaml}"

python3 -m src.export_onnx --config "$CONFIG_PATH"
