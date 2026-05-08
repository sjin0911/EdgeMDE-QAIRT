# QAIRT 온디바이스 포팅 및 성능 분석 가이드

이 문서는 **Depth Anything V2** 모델을 Qualcomm DSP(HTP)로 포팅하고, 성능(Latency) 및 정확도(MSE)를 분석하는 전체 과정과 자동화 스크립트 사용법을 설명합니다.

---

## 1. 환경 구성 (Environment)

### Docker 컨테이너 실행
프로젝트 코드가 담긴 로컬 폴더를 컨테이너 내부의 `/workspace`에 연결하고, 호스트의 ADB 연결을 공유하여 실행합니다.

```bash
# 로컬 작업 폴더를 마운트하고 네트워크를 공유하여 실행
docker run -itd --name edgemde-dev-container \
  --net host \
  -v $(pwd):/workspace \
  edgemde-qairt bash
```

*   **`-v $(pwd):/workspace`**: 로컬의 코드를 컨테이너와 동기화합니다.
*   **`--net host`**: 호스트 PC에 연결된 ADB 디바이스를 컨테이너 내에서도 그대로 인식하게 합니다.

---

## 2. 모델 분석 도구 (Utility)

모델을 수정하거나 해상도를 변경한 후, QAIRT에서 지원하지 않는 연산자(OP)가 있는지 확인하는 도구입니다.

```bash
# 도커 내부에서 실행
python3 src/check_onnx_ops.py [모델경로.onnx]
```
*   가장 많이 사용된 OP 순서대로 통계를 출력하여 병목 구간을 예측하는 데 도움을 줍니다.

---

## 3. 성능 분석 실행 (Profiling)

모든 실행 과정은 프로파일링 결과(CSV) 생성까지 자동화되어 있습니다.

### A. Device DSP 실행 및 분석
타겟 디바이스(보드/폰)에서 DSP 가속을 사용하여 실행하고 결과를 가져옵니다.

```bash
# 호스트 PC 터미널에서 실행
./scripts/run_dsp_profiling.sh
```
*   **자동화 내용:** ADB Push → DSP 실행(Detailed Profiling) → 결과 Pull → CSV 변환

### B. Host CPU 실행 및 분석
도커 내부(x86 CPU) 환경에서 QAIRT CPU 백엔드를 사용하여 실행합니다.

```bash
# 도커 내부 터미널에서 실행
./scripts/run_cpu_profiling.sh
```
*   **자동화 내용:** CPU 실행(Detailed Profiling) → CSV 변환

---

## 4. 결과 분석 방법 (Analysis)

### 추론 결과물 확인
실행이 완료되면 아래 경로에서 결과물(`.raw`)을 확인할 수 있습니다.
*   **DSP 결과:** `data/outputs/device_dsp/depth_anything_v2_vits_quick_gelu_bias8/Result_0/output.raw`
*   **CPU 결과:** `data/outputs/device_cpu/depth_anything_v2_vits_quick_gelu_bias8/Result_0/output.raw`

### 성능(속도) 확인
각 결과 폴더 내의 **`profiling_result.csv`** 파일을 확인합니다.

1.  **전체 추론 시간:** `Message`가 `EXECUTE`, `Timing Source`가 `BACKEND`, `Event Level`이 `ROOT`인 행의 `Time` 값을 봅니다. (단위: US)
2.  **레이어별 시간:** `Event Level`이 `SUB-EVENT`인 행들을 분석하여 어떤 연산자가 병목(Bottle-neck)인지 파악합니다.

---

## 5. 주요 설정 변경 (해상도 등)

입력 해상도를 변경하여 다시 포팅하고 싶을 경우 다음 순서를 따릅니다.

1.  **`configs/depth_model.yaml`** 수정: `input: shape` 값을 변경 (패치 크기인 **14의 배수** 권장).
2.  **ONNX 다시 생성:** `python3 src/export_onnx.py --config configs/depth_model.yaml` 실행.
3.  **DLC 변환 및 양자화:** (기존 QAIRT 변환 프로세스 수행)
4.  **성능 재측정:** 위 3번 항목의 스크립트들을 다시 실행하여 속도 개선 효과 확인.

---

## 6. 문제 해결 (Troubleshooting)

*   **`Read-only file system` 에러:** `adb shell` 내부에서 `/data/local/tmp` 이외의 경로에 폴더를 만들려 할 때 발생합니다. 작업 경로는 항상 `/data/local/tmp` 하위로 설정하세요.
*   **`Result_0` 폴더가 안 생기는 경우:** `input_list_host.txt` 등 입력 리스트 파일 내의 데이터 경로가 실제 도커 내부 경로와 일치하는지 확인하세요.
*   **`command not found: qnn-profile-viewer`:** 스크립트 상단의 `QAIRT_ROOT` 경로가 현재 설치된 버전과 일치하는지 확인하세요.
