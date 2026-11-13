"""2D segmentation model (PyTorch): U-Net++ on an ImageNet-pretrained encoder.

Input: 3 x H x W (three CT windows). Output: 1 x H x W logits (clot/haemorrhage yes/no).
The same model is used for edema or midline masks once such data is available (--target).
"""

import numpy as np
import torch
import torch.nn as nn


class SegModel(nn.Module):
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(self, arch="unetplusplus", encoder="efficientnet-b3", pretrained=True):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.net = smp.create_model(arch, encoder_name=encoder, encoder_weights="imagenet" if pretrained else None,
                                    in_channels=3, classes=1)
        self.register_buffer("mean", self.MEAN.clone(), persistent=False)
        self.register_buffer("std", self.STD.clone(), persistent=False)

    def forward(self, x):                      # x in [0, 1]
        return self.net((x - self.mean) / self.std)


def load_checkpoint(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = SegModel(ck["arch"], ck["encoder"], pretrained=False)
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck


@torch.no_grad()
def predict_slices(model, images_u8, device, batch=16, tta=True):
    """images_u8: (N, 3, H, W) uint8 -> probabilities (N, H, W). tta = average with the mirrored slice."""
    out = []
    for i in range(0, len(images_u8), batch):
        x = torch.from_numpy(images_u8[i:i + batch]).float().div(255).to(device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            p = torch.sigmoid(model(x).float())
            if tta:
                p = (p + torch.sigmoid(model(x.flip(-1)).float()).flip(-1)) / 2
        out.append(p[:, 0].cpu().numpy())
    return np.concatenate(out)
