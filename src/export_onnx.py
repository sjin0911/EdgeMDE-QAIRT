from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml

from src.models import SimpleCNN
from src.models.depth_anything_v2_vits import DepthAnythingV2Wrapper

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "depth_model.yaml"

MODEL_REGISTRY = {
    "simple_cnn": SimpleCNN,
    "depth_anything_v2_vits": DepthAnythingV2Wrapper,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a PyTorch model to ONNX and validate it with ONNX Runtime."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML config for the export job.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return config


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def build_model(config: dict[str, Any]) -> torch.nn.Module:
    model_name = config["model"]["name"]
    model_params = config["model"].get("params", {})

    try:
        model_cls = MODEL_REGISTRY[model_name]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unsupported model '{model_name}'. Available models: {available}"
        ) from exc

    return model_cls(**model_params).eval()


def create_dummy_input(config: dict[str, Any]) -> torch.Tensor:
    input_config = config["input"]
    shape = input_config["shape"]
    seed = input_config.get("seed", 0)

    torch.manual_seed(seed)
    return torch.randn(*shape)


def export_model(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    config: dict[str, Any],
) -> tuple[Path, torch.Tensor]:
    export_config = config["export"]
    output_path = resolve_project_path(export_config["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        torch_output = model(dummy_input)

    export_kwargs = {
        "input_names": export_config["input_names"],
        "output_names": export_config["output_names"],
        "opset_version": export_config["opset_version"],
        "dynamic_axes": None,
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = export_config.get("dynamo", False)

    torch.onnx.export(
        model,
        (dummy_input,),
        str(output_path),
        **export_kwargs,
    )

    return output_path, torch_output


def validate_export(
    onnx_path: Path,
    dummy_input: torch.Tensor,
    torch_output: torch.Tensor,
    config: dict[str, Any],
) -> None:
    validation_config = config["validation"]
    input_name = config["export"]["input_names"][0]

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    ort_session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    ort_output = ort_session.run(
        None,
        {input_name: dummy_input.detach().cpu().numpy()},
    )[0]

    # rtol, atol 값을 float으로 강제 변환 (YAML 파서의 타입 이슈 방지)
    rtol = float(validation_config["rtol"])
    atol = float(validation_config["atol"])

    # 모델 출력이 multiple outputs(list/tuple)인 경우 첫 번째 출력만 비교
    actual_torch_output = torch_output[0] if isinstance(torch_output, (list, tuple)) else torch_output

    np.testing.assert_allclose(
        actual_torch_output.detach().cpu().numpy(),
        ort_output,
        rtol=rtol,
        atol=atol,
    )


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)

    model = build_model(config)
    dummy_input = create_dummy_input(config)
    onnx_path, torch_output = export_model(model, dummy_input, config)
    validate_export(onnx_path, dummy_input, torch_output, config)

    print(f"Config loaded: {config_path}")
    print(f"ONNX export complete: {onnx_path}")
    print("ONNX structural validation passed")
    print("PyTorch vs ONNX Runtime output validation passed")


if __name__ == "__main__":
    main()
