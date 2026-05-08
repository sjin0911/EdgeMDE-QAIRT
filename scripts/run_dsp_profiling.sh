#!/bin/bash
#
# Galaxy S20 (SM8250) DSP Profiling Script
# Pushes the latest DLC and input, runs profiling, and pulls results.
#

set -e

# 1. Path Setup
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QAIRT_ROOT="/workspace/qairt/2.44.0.260225"
export PATH="$QAIRT_ROOT/bin/x86_64-linux-clang:$PATH"

# Device configuration
export TARGET=/data/local/tmp/depth_anything_v2_vits_dsp

# Model Configuration (Matching your 392_base experiment)
RES="252"
DLC_NAME="depth_anything_v2_vits_${RES}_base_quantized_bias8.dlc"
LOCAL_DLC="${ROOT_DIR}/models/dlc/${DLC_NAME}"
LOCAL_RAW="${ROOT_DIR}/data/inputs_raw_fmt/depth_anything_v2_vits/sample_1x3x${RES}x${RES}_float32.raw"

# Output configuration
OUTPUT_DIR_NAME="output_${RES}_base"
HOST_OUTPUT_BASE="${ROOT_DIR}/data/outputs/device_dsp/depth_anything_v2_vits_${RES}_base"

echo "========================================================"
echo " STEP 1: Pushing artifacts to device"
echo "========================================================"
echo "DLC: ${DLC_NAME}"
echo "Input: $(basename ${LOCAL_RAW})"

# Push DLC
adb push "${LOCAL_DLC}" "${TARGET}/"

# Push Raw Input
adb push "${LOCAL_RAW}" "${TARGET}/"

# Generate and push Target Input List
TARGET_LIST="${ROOT_DIR}/data/inputs_raw_fmt/depth_anything_v2_vits/input_list_target_dsp.txt"
echo "$(basename ${LOCAL_RAW})" > "${TARGET_LIST}"
adb push "${TARGET_LIST}" "${TARGET}/input_list_target_dsp.txt"

echo ""
echo "========================================================"
echo " STEP 2: Running QNN NPU Profiling on Device"
echo "========================================================"

adb shell "cd ${TARGET} && \
  rm -rf ${OUTPUT_DIR_NAME} && mkdir -p ${OUTPUT_DIR_NAME} && \
  export LD_LIBRARY_PATH=${TARGET} && \
  export ADSP_LIBRARY_PATH='${TARGET}/dsp;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp' && \
  ./qnn-net-run \
    --backend libQnnDsp.so \
    --profiling_level detailed \
    --model libQnnModelDlc.so \
    --dlc_path ${DLC_NAME} \
    --input_list input_list_target_dsp.txt \
    --output_dir ${OUTPUT_DIR_NAME} \
    --log_level verbose"

echo ""
echo "========================================================"
echo " STEP 3: Pulling results and generating report"
echo "========================================================"

mkdir -p "${HOST_OUTPUT_BASE}"
adb pull "${TARGET}/${OUTPUT_DIR_NAME}" "${HOST_OUTPUT_BASE}/"

# Convert profiling data to CSV
qnn-profile-viewer \
    --input_log "${HOST_OUTPUT_BASE}/${OUTPUT_DIR_NAME}/qnn-profiling-data"*".log" \
    --output "${HOST_OUTPUT_BASE}/${OUTPUT_DIR_NAME}/profiling_result.csv"

# --- Metrics Extraction ---
CSV_FILE="${HOST_OUTPUT_BASE}/${OUTPUT_DIR_NAME}/profiling_result.csv"

if [ -f "$CSV_FILE" ]; then
    # 1. QNN Execute Time (us -> ms)
    EXEC_TIME_US=$(grep "QNN (execute) time" "$CSV_FILE" | cut -d',' -f3 | head -n 1)
    EXEC_TIME_MS=$(awk "BEGIN {printf \"%.2f\", $EXEC_TIME_US / 1000}")

    # 2. IPS (Inferences Per Second)
    IPS=$(grep "EXECUTE IPS" "$CSV_FILE" | cut -d',' -f3 | head -n 1)

    # 3. MatMul_1 Average (us -> ms)
    MATMUL_1_AVG_US=$(grep "attn/MatMul_1" "$CSV_FILE" | awk -F',' '{sum+=$3; count++} END {if (count > 0) print sum/count; else print 0}')
    MATMUL_1_AVG_MS=$(awk "BEGIN {printf \"%.2f\", $MATMUL_1_AVG_US / 1000}")

    echo ""
    echo "========================================================"
    echo " 📊 PERFORMANCE SUMMARY (DSP)"
    echo "========================================================"
    echo " Total Execute Time : ${EXEC_TIME_MS} ms"
    echo " Inferences Per Sec : ${IPS} IPS"
    echo " MatMul_1 Avg Time  : ${MATMUL_1_AVG_MS} ms"
    echo "--------------------------------------------------------"
    echo " Results Folder     : ${HOST_OUTPUT_BASE}/${OUTPUT_DIR_NAME}"
    echo " CSV Report         : ${CSV_FILE}"
    echo "========================================================"
else
    echo "Error: Profiling CSV not found at $CSV_FILE"
fi