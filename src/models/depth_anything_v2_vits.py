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
            
            # Interpolate pos_embed if shapes mismatch
            if 'pretrained.pos_embed' in state_dict:
                pos_embed_checkpoint = state_dict['pretrained.pos_embed']
                embedding_size = pos_embed_checkpoint.shape[-1]
                num_patches_model = self.model.pretrained.patch_embed.num_patches
                num_extra_tokens = self.model.pretrained.num_tokens
                
                if pos_embed_checkpoint.shape[1] != num_patches_model + num_extra_tokens:
                    print(f"[INFO] Interpolating pos_embed from {pos_embed_checkpoint.shape[1]} to {num_patches_model + num_extra_tokens}")
                    
                    # Separate cls_token and patch_embeddings
                    pos_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                    patch_pos_embed = pos_embed_checkpoint[:, num_extra_tokens:]
                    
                    # Calculate grid sizes
                    orig_size = int((pos_embed_checkpoint.shape[1] - num_extra_tokens) ** 0.5)
                    new_size = int(num_patches_model ** 0.5)
                    
                    # Interpolate
                    patch_pos_embed = patch_pos_embed.reshape(1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                    patch_pos_embed = torch.nn.functional.interpolate(
                        patch_pos_embed, 
                        size=(new_size, new_size), 
                        mode='bicubic', 
                        align_corners=False
                    )
                    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
                    
                    # Concatenate back
                    new_pos_embed = torch.cat((pos_tokens, patch_pos_embed), dim=1)
                    state_dict['pretrained.pos_embed'] = new_pos_embed
            
            self.model.load_state_dict(state_dict)

    def forward(self, x):
        return self.model(x)
