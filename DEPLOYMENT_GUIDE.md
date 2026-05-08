# Depth Anything V2 DSP Deployment Guide (QAIRT)

이 문서는 Depth Anything V2 모델을 Qualcomm DSP(Hexagon) 환경으로 최적화하여 배포하고 성능을 측정하는 전체 과정을 설명합니다.

## 🚀 전체 파이프라인 요약

모델 배포는 크게 **[데이터 준비 -> 모델 추출 -> DLC 변환 -> 성능 측정 -> 시각화]**의 5단계로 진행됩니다.

---

## 1. 입력 데이터 준비 (Image to Raw)
양자화(Quantization) 및 테스트에 사용할 실제 이미지 데이터를 모델 해상도에 맞춰 변환합니다.

```bash
# 예: 252x252 해상도로 sample.jpg 변환
python3 src/prepare_input.py --res 252 --input data/raw/sample.jpg
```
*   **결과:** `data/inputs_raw_fmt/depth_anything_v2_vits/` 폴더에 `.raw` 파일 및 `input_list_host.txt` 생성.

## 2. 모델 추출 (PyTorch to ONNX)
설정 파일(`yaml`)에 명시된 해상도와 구조로 ONNX 모델을 추출합니다.

1.  `configs/depth_model_dsp_quick_gelu.yaml`에서 `shape`를 원하는 해상도로 수정 (예: `[1, 3, 252, 252]`).
2.  아래 명령어 실행:
```bash
python3 src/export_onnx.py --config configs/depth_model_dsp_quick_gelu.yaml
```
*   **주의:** 해상도는 반드시 **14의 배수**여야 합니다 (예: 252, 266, 280, 336, 392, 518).

## 3. DLC 변환 및 양자화 (ONNX to DLC)
ONNX 모델을 QNN/SNPE에서 사용하는 DLC 형식으로 변환하고 DSP 실행을 위해 8-bit 양자화를 수행합니다.

1.  `scripts/prepare_dsp_dlc.sh`를 열어 `INPUT_RES` 변수를 위에서 설정한 해상도와 동일하게 수정합니다.
2.  스크립트 실행:
```bash
./scripts/prepare_dsp_dlc.sh
```
*   이 과정에서 1단계에서 준비한 실제 이미지를 사용하여 가중치를 양자화(Calibration)합니다.

## 4. 디바이스 프로파일링 (DSP Execution)
변환된 모델을 실제 안드로이드 기기의 DSP에서 실행하고 성능(Latency, IPS)을 측정합니다.

1.  `scripts/run_dsp_profiling.sh`를 열어 `RES` 변수를 수정합니다.
2.  스크립트 실행:
```bash
./scripts/run_dsp_profiling.sh
```
*   **자동 분석:** 실행이 끝나면 터미널에 **Total Execute Time**, **IPS**, **MatMul_1 Average** 지표가 요약되어 출력됩니다.

## 5. 결과 시각화 (Raw to PNG)
DSP에서 추론된 바이너리 결과값(.raw)을 사람이 볼 수 있는 뎁스 맵 이미지로 변환합니다.

```bash
# 해상도는 파일 크기를 보고 자동으로 판별됩니다.
python3 src/visualize_output.py --raw [결과_파일_경로]/output.raw
```

---

## 💡 주요 팁 및 해결 방법

### 1. 결과가 뭉개져서(노이즈처럼) 보일 때
*   양자화 과정에서 실제 이미지가 아닌 랜덤 노이즈가 들어갔을 가능성이 높습니다.
*   `python3 src/prepare_input.py`를 먼저 실행했는지 확인하세요.

### 2. "Unable to broadcast shapes" 에러 발생 시
*   모델의 `pos_embed` 크기와 입력 해상도가 일치하지 않을 때 발생합니다.
*   `dinov2.py`가 기본값인 **518**로 고정되어 있는지 확인하고, ONNX를 다시 추출하세요.

### 3. 성능이 너무 느릴 때 (예: 40초 이상)
*   프로파일링 요약 리포트에서 `MatMul_1 Avg Time`을 확인하세요.
*   특정 연산이 DSP 가속을 받지 못하고 CPU로 폴백될 경우 발생하며, 이 경우 모델 구조 변경이 필요할 수 있습니다.
