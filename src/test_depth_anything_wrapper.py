import torch
from src.models.depth_anything_v2_vits import DepthAnythingV2Wrapper

model = DepthAnythingV2Wrapper(
    encoder="vits",
    checkpoint_path="models/pytorch/depth_anything_v2_vits.pth"
)
model.eval()

x = torch.randn(1, 3, 518, 518)

with torch.no_grad():
    y = model(x)

print(type(y))
if isinstance(y, (list, tuple)):
    print(len(y))
    for i, t in enumerate(y):
        print(i, t.shape)
else:
    print(y.shape)