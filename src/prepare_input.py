import cv2
import torch
import numpy as np
from pathlib import Path
from torchvision.transforms import Compose
import sys

# Depth Anything V2 라이브러리 경로 추가
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT / "external" / "Depth-Anything-V2"
sys.path.append(str(REPO_ROOT))

# util.transform 임포트를 위해 추가 경로 설정이 필요할 수 있음
sys.path.append(str(REPO_ROOT / "depth_anything_v2"))
from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet

import argparse

def main():
    parser = argparse.ArgumentParser(description="Prepare real image input for Depth Anything V2")
    parser.add_argument("--res", "-r", type=int, default=518, help="Target resolution (e.g., 518, 392, 336)")
    parser.add_argument("--input", "-i", type=str, default="data/raw/sample.jpg", help="Path to input image")
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input
    output_res = args.res
    output_dir = PROJECT_ROOT / "data/inputs_raw_fmt/depth_anything_v2_vits"
    output_path = output_dir / f"sample_1x3x{output_res}x{output_res}_float32.raw"
    
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 이미지 로드
    if not input_path.exists():
        print(f"Error: {input_path}를 찾을 수 없습니다.")
        return
        
    raw_image = cv2.imread(str(input_path))
    if raw_image is None:
        print(f"Error: {input_path} 로드 실패.")
        return

    print(f"[INFO] Processing {input_path} to {output_res}x{output_res}...")

    # 2. 전처리 파이프라인 (모델과 동일하게 설정)
    transform = Compose([
        Resize(
            width=output_res,
            height=output_res,
            resize_target=False,
            keep_aspect_ratio=False, 
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    
    # 3. 변환 수행
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
    image = transform({'image': image})['image'] # [3, H, W]
    
    # 4. 바이너리로 저장
    image.tofile(output_path)
    print(f"변환 성공: {output_path} (Size: {image.nbytes} bytes)")
    
    # 5. input_list_host.txt 갱신
    list_path = output_dir / "input_list_host.txt"
    with open(list_path, "w") as f:
        # We use the /workspace/ path convention for QAIRT compatibility inside Docker
        f.write(f"/workspace/data/inputs_raw_fmt/depth_anything_v2_vits/sample_1x3x{output_res}x{output_res}_float32.raw\n")
    print(f"리스트 갱신 완료: {list_path}")

if __name__ == "__main__":
    main()
