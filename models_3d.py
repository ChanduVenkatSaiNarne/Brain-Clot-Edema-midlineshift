"""3D segmentation models (MONAI). Any number of input channels (MRI sequences or CT) and outputs.

  segresnet  SegResNet (Myronenko 2018; basis of MONAI's BraTS 2021 runner-up). Residual CNN, ~4.7 M params.
             Best accuracy per GB of GPU memory. Default patch 128^3.
  swinunetr  Swin UNETR (Hatamizadeh 2022): Swin-transformer encoder + CNN decoder, ~62 M params.
             The most modern model here, and the most memory-hungry. Default patch 96^3.
  dynunet    DynUNet: nnU-Net-style residual 3D U-Net with instance norm, ~16-30 M params. Patch 128^3.
             Strides adapt to flat CT patches (no downsampling along thin axes).

For the strongest baseline also train real nnU-Net v2 on the same converted dataset (see README).
"""

import inspect

DEFAULT_CUBE = {"segresnet": 128, "swinunetr": 96, "dynunet": 128}


def _dyn_strides(roi, levels=5):
    strides, size = [[1, 1, 1]], list(roi)
    for _ in range(levels):
        s = [2 if v >= 8 else 1 for v in size]
        strides.append(s)
        size = [v // st for v, st in zip(size, s)]
    return strides


def build_model(name, in_channels, out_channels, roi):
    from monai.networks import nets

    name = name.lower()
    if name == "segresnet":
        return nets.SegResNet(blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1], init_filters=16,
                              in_channels=in_channels, out_channels=out_channels, dropout_prob=0.2)
    if name == "swinunetr":
        kw = dict(in_channels=in_channels, out_channels=out_channels, feature_size=48, use_checkpoint=True)
        if "img_size" in inspect.signature(nets.SwinUNETR.__init__).parameters:   # older MONAI versions
            kw["img_size"] = tuple(roi)
        return nets.SwinUNETR(**kw)
    if name == "dynunet":
        strides = _dyn_strides(roi)
        return nets.DynUNet(spatial_dims=3, in_channels=in_channels, out_channels=out_channels,
                            kernel_size=[[3, 3, 3]] * len(strides), strides=strides,
                            upsample_kernel_size=strides[1:], norm_name="instance",
                            deep_supervision=False, res_block=True)
    raise ValueError(f"Unknown model '{name}'. Choose from {list(DEFAULT_CUBE)}")
