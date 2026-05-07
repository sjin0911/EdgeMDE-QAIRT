#!/bin/bash

# QAIRT 경로 설정
QAIRT_ROOT="/workspace/qairt/2.44.0.260225"
export PATH="$QAIRT_ROOT/bin/x86_64-linux-clang:$PATH"

# 타겟 디바이스 내의 작업 경로 설정
export TARGET=/data/local/tmp/depth_anything_v2_vits_dsp

echo "====================================="
echo "Running QNN NPU Profiling on Device"
echo "Target Path: $TARGET"
echo "====================================="

adb shell "cd $TARGET && \
  rm -rf output_quick_gelu_bias8 && mkdir -p output_quick_gelu_bias8 && \
  export LD_LIBRARY_PATH=$TARGET && \
  export ADSP_LIBRARY_PATH='$TARGET/dsp;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp' && \
  ./qnn-net-run \
    --backend libQnnDsp.so \
    --profiling_level detailed \
    --model libQnnModelDlc.so \
    --dlc_path depth_anything_v2_vits_518_dsp_gatherfix_quick_gelu_quantized_bias8.dlc \
    --input_list input_list_target_dsp.txt \
    --output_dir output_quick_gelu_bias8 \
    --log_level verbose"

echo "====================================="
echo "Execution Finished! Pulling results..."
echo "====================================="

# 결과 가져오기
OUTPUT_BASE="./data/outputs/device_dsp/depth_anything_v2_vits_quick_gelu_bias8"
adb pull $TARGET/output_quick_gelu_bias8 $OUTPUT_BASE
echo "Done! Results saved to $OUTPUT_BASE"

# 프로파일링 결과 변환 (CSV)
qnn-profile-viewer \
    --input_log $OUTPUT_BASE/output_quick_gelu_bias8/qnn-profiling-data*.log \
    --output $OUTPUT_BASE/output_quick_gelu_bias8/profiling_result.csv

echo "Profiling CSV generated at $OUTPUT_BASE/output_quick_gelu_bias8/profiling_result.csv"