import numpy as np
import cv2
import argparse
from pathlib import Path

def visualize_raw(raw_path, width, height, output_path):
    # 1. Raw 파일 로드 (float32)
    data = np.fromfile(raw_path, dtype=np.float32)
    
    # 2. Reshape (Depth Anything V2의 경우 HxW 형태)
    try:
        depth = data.reshape((height, width))
    except ValueError:
        print(f"Error: 데이터 크기({len(data)})가 해상도({width}x{height}={width*height})와 맞지 않습니다.")
        return

    # 3. 정규화 (0~255)
    depth_min = depth.min()
    depth_max = depth.max()
    depth_norm = (depth - depth_min) / (depth_max - depth_min + 1e-6)
    depth_norm = (depth_norm * 255).astype(np.uint8)

    # 4. 컬러맵 적용 (Depth Map 시각화에 좋은 MAGMA/INFERNO 스타일)
    color_depth = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)

    # 5. 저장
    cv2.imwrite(str(output_path), color_depth)
    print(f"시각화 완료: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True, help="Path to .raw file")
    parser.add_argument("--res", type=int, default=392, help="Resolution (width and height)")
    parser.add_argument("--output", type=Path, help="Output image path")
    args = parser.parse_args()

    out_path = args.output if args.output else args.raw.with_suffix(".png")
    visualize_raw(args.raw, args.res, args.res, out_path)
