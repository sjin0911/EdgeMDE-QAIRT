import torch
import torch.nn as nn
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2] / "external" / "Depth-Anything-V2"
sys.path.append(str(REPO_ROOT))

from depth_anything_v2.dpt import DepthAnythingV2


class DepthAnythingV2Wrapper(nn.Module):
    def __init__(self, encoder="vits", checkpoint_path=None, dinov2_act_layer="gelu"):
        super().__init__()

        model_configs = {
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        }

        self.model = DepthAnythingV2(
            **model_configs[encoder],
            dinov2_act_layer=dinov2_act_layer,
        )

        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(state_dict)

    def forward(self, x):
        return self.model(x)
