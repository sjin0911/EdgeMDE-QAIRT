import argparse
import onnx
from collections import Counter
import sys

def main():
    parser = argparse.ArgumentParser(description="Count and print all OP types in an ONNX model.")
    parser.add_argument("onnx_path", type=str, help="Path to the ONNX model file")
    args = parser.parse_args()

    try:
        print(f"Loading model: {args.onnx_path}")
        model = onnx.load(args.onnx_path)
    except Exception as e:
        print(f"Failed to load ONNX model: {e}")
        sys.exit(1)

    # 모든 노드의 op_type을 세어서 빈도수 측정
    ops = Counter(node.op_type for node in model.graph.node)

    print("\n[ ONNX Model OP Types Summary ]")
    print(f"{'OP Type':<20} | {'Count':<5}")
    print("-" * 30)
    
    total_ops = 0
    for op, count in ops.most_common():
        print(f"{op:<20} | {count:<5}")
        total_ops += count
        
    print("-" * 30)
    print(f"{'Total Nodes':<20} | {total_ops:<5}")

if __name__ == "__main__":
    main()
