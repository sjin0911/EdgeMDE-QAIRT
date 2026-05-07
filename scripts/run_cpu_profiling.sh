#!/bin/bash

# QAIRT 경로 설정
QAIRT_ROOT="/workspace/qairt/2.44.0.260225"
export PATH="$QAIRT_ROOT/bin/x86_64-linux-clang:$PATH"

echo "====================================="
echo "Running QNN CPU Profiling on Host (Docker)"
echo "====================================="

# 기존 출력 폴더가 있다면 삭제 후 재생성 (깔끔한 결과 저장을 위해)
OUTPUT_DIR="/workspace/data/outputs/device_cpu/depth_anything_v2_vits_quick_gelu_bias8"
rm -rf $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

qnn-net-run \
  --backend $QAIRT_ROOT/lib/x86_64-linux-clang/libQnnCpu.so \
  --model $QAIRT_ROOT/lib/x86_64-linux-clang/libQnnModelDlc.so \
  --dlc_path /workspace/models/dlc/depth_anything_v2_vits_518_dsp_gatherfix_quick_gelu_quantized_bias8.dlc \
  --input_list /workspace/data/inputs_raw_fmt/depth_anything_v2_vits/input_list_host.txt \
  --output_dir $OUTPUT_DIR \
  --profiling_level detailed \
  --log_level warn

echo "====================================="
echo "Execution Finished! Generating profiling report..."
echo "====================================="

# 프로파일링 결과 변환 (CSV)
qnn-profile-viewer \
    --input_log $OUTPUT_DIR/qnn-profiling-data.log \
    --output $OUTPUT_DIR/profiling_result.csv

echo "Done! CPU Profiling Results saved to: $OUTPUT_DIR"
