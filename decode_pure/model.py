"""
Pure-PyTorch implementation of DECODE's SigmaMUNet (DoubleMUnet + UNet2d + MLTHeads).
Matches decode==0.10.2 state_dict keys and tensor shapes bit-for-bit so checkpoints
(model_best.pt / model_final.pt) load with strict=True in any modern PyTorch version.
"""

import warnings
from pathlib import Path
from typing import Union
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_activation(activation):
    if activation is None:
        return None
    if isinstance(activation, str):
        if activation.startswith("nn."):
            activation = activation[3:]
        return getattr(nn, activation)()
    if isinstance(activation, nn.Module):
        return activation
    act = activation()
    assert isinstance(act, nn.Module)
    return act


class Upsample(nn.Module):
    def __init__(self, scale_factor, mode="nearest", in_channels=None, out_channels=None, align_corners=False, ndim=2):
        super().__init__()
        self.mode = mode
        self.scale_factor = scale_factor
        self.align_corners = align_corners
        if in_channels != out_channels:
            if ndim == 2:
                self.conv = nn.Conv2d(in_channels, out_channels, 1)
            elif ndim == 3:
                self.conv = nn.Conv3d(in_channels, out_channels, 1)
            else:
                raise ValueError("Only 2d and 3d supported")
        else:
            self.conv = None

    def forward(self, input_t):
        x = F.interpolate(
            input_t,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )
        if self.conv is not None:
            return self.conv(x)
        return x


class UNet2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        depth=4,
        initial_features=64,
        gain=2,
        pad_convs=False,
        norm=None,
        norm_groups=None,
        p_dropout=None,
        final_activation=None,
        activation=nn.ReLU(),
        pool_mode="StrideConv",
        skip_gn_level=None,
        upsample_mode="bilinear",
    ):
        super().__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pad_convs = pad_convs
        self.pool_mode = pool_mode
        self.norm = norm
        self.norm_groups = norm_groups
        self.p_dropout = p_dropout
        self.skip_gn_level = skip_gn_level

        n_features = [in_channels] + [initial_features * gain ** level for level in range(self.depth)]
        self.features_per_level = n_features
        self.encoder = nn.ModuleList(
            [
                self._conv_block(n_features[level], n_features[level + 1], level, part="encoder", activation=activation)
                for level in range(self.depth)
            ]
        )
        self.base = self._conv_block(n_features[-1], gain * n_features[-1], part="base", level=depth, activation=activation)

        dec_features = [initial_features * gain ** level for level in range(self.depth + 1)][::-1]
        self.decoder = nn.ModuleList(
            [
                self._conv_block(dec_features[level], dec_features[level + 1], self.depth - level - 1, part="decoder", activation=activation)
                for level in range(self.depth)
            ]
        )
        self.poolers = nn.ModuleList([self._pooler(level + 1) for level in range(self.depth)])
        self.upsamplers = nn.ModuleList(
            [
                self._upsampler(dec_features[level], dec_features[level + 1], self.depth - level - 1, mode=upsample_mode)
                for level in range(self.depth)
            ]
        )
        self.out_conv = nn.Conv2d(dec_features[-1], out_channels, 1)
        self.activation = _get_activation(final_activation)

    def _conv_block(self, in_channels, out_channels, level, part, activation=nn.ReLU()):
        padding = 1 if self.pad_convs else 0
        if self.norm is not None:
            num_groups1 = min(in_channels, self.norm_groups)
            num_groups2 = min(out_channels, self.norm_groups)
        else:
            num_groups1, num_groups2 = None, None

        if self.norm is None or (self.skip_gn_level is not None and self.skip_gn_level >= level):
            sequence = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding),
                activation,
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding),
                activation,
            )
        elif self.norm == "GroupNorm":
            sequence = nn.Sequential(
                nn.GroupNorm(num_groups1, in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding),
                activation,
                nn.GroupNorm(num_groups2, out_channels),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding),
                activation,
            )
        else:
            raise ValueError(f"Unsupported norm: {self.norm}")

        if self.p_dropout is not None:
            sequence.add_module("droupout", nn.Dropout2d(p=self.p_dropout))
        return sequence

    def _upsampler(self, in_channels, out_channels, level, mode):
        return Upsample(
            in_channels=in_channels,
            out_channels=out_channels,
            scale_factor=2,
            mode=mode,
            ndim=2,
            align_corners=False if mode == "bilinear" else None,
        )

    def _pooler(self, level):
        if self.pool_mode == "MaxPool":
            return nn.MaxPool2d(2)
        if self.pool_mode == "StrideConv":
            return nn.Conv2d(
                self.features_per_level[level],
                self.features_per_level[level],
                kernel_size=2,
                stride=2,
                padding=0,
            )
        raise ValueError(f"Unsupported pool_mode: {self.pool_mode}")

    @staticmethod
    def _crop_tensor(input_, shape_to_crop):
        input_shape = input_.shape
        shape_diff = tuple((ish - csh) // 2 for ish, csh in zip(input_shape, shape_to_crop))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if all(sd == 0 for sd in shape_diff):
                return input_
        crop = tuple(slice(sd, sh - sd) for sd, sh in zip(shape_diff, input_shape))
        return input_[crop]

    def _crop_and_concat(self, from_decoder, from_encoder):
        cropped = self._crop_tensor(from_encoder, from_decoder.shape)
        return torch.cat((cropped, from_decoder), dim=1)

    def forward(self, input_t):
        x = input_t
        encoder_out = []
        for level in range(self.depth):
            x = self.encoder[level](x)
            encoder_out.append(x)
            x = self.poolers[level](x)
        x = self.base(x)
        encoder_out = encoder_out[::-1]
        for level in range(self.depth):
            x = self.upsamplers[level](x)
            x = self.decoder[level](self._crop_and_concat(x, encoder_out[level]))
        x = self.out_conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class MLTHeads(nn.Module):
    def __init__(self, in_channels, out_channels, last_kernel, norm, norm_groups, padding, activation):
        super().__init__()
        self.norm = norm
        self.norm_groups = norm_groups
        groups_1 = min(in_channels, self.norm_groups) if self.norm is not None else None
        if norm == "GroupNorm":
            self.core = nn.Sequential(
                nn.GroupNorm(groups_1, in_channels),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=padding),
                activation,
            )
        elif norm is None:
            self.core = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=padding),
                activation,
            )
        else:
            raise NotImplementedError(f"Unsupported head norm: {norm}")
        self.out_conv = nn.Conv2d(in_channels, out_channels, kernel_size=last_kernel, padding=0)

    def forward(self, x):
        return self.out_conv(self.core(x))


class SigmaMUNet(nn.Module):
    """Exact pure-PyTorch mirror of decode.neuralfitter.models.SigmaMUNet."""

    ch_out = 10
    out_channels_heads = (1, 4, 4, 1)
    sigmoid_ch_ix = [0, 1, 5, 6, 7, 8, 9]
    tanh_ch_ix = [2, 3, 4]
    pxyz_sig_ch_ix = slice(5, 9)
    sigma_eps_default = 0.001

    def __init__(
        self,
        ch_in: int = 3,
        *,
        depth_shared: int = 2,
        depth_union: int = 2,
        initial_features: int = 48,
        inter_features: int = 48,
        norm=None,
        norm_groups=None,
        norm_head=None,
        norm_head_groups=None,
        pool_mode: str = "StrideConv",
        upsample_mode: str = "nearest",
        skip_gn_level: Union[None, bool] = None,
        activation=nn.ReLU(),
        init_custom: bool = True,
    ):
        super().__init__()
        assert ch_in in (1, 3), "ch_in must be 1 or 3"
        self.ch_in = ch_in
        self.unet_shared = UNet2d(
            in_channels=1,
            out_channels=inter_features,
            depth=depth_shared,
            initial_features=initial_features,
            pad_convs=True,
            norm=norm,
            norm_groups=norm_groups,
            skip_gn_level=skip_gn_level,
            pool_mode=pool_mode,
            upsample_mode=upsample_mode,
            activation=activation,
        )
        self.unet_union = UNet2d(
            in_channels=inter_features * ch_in,
            out_channels=inter_features,
            depth=depth_union,
            initial_features=initial_features,
            pad_convs=True,
            norm=norm,
            norm_groups=norm_groups,
            skip_gn_level=skip_gn_level,
            pool_mode=pool_mode,
            upsample_mode=upsample_mode,
            activation=activation,
        )
        self.mt_heads = nn.ModuleList(
            [
                MLTHeads(
                    in_channels=inter_features,
                    out_channels=ch_o,
                    activation=activation,
                    last_kernel=1,
                    padding=1,
                    norm=norm_head,
                    norm_groups=norm_head_groups,
                )
                for ch_o in self.out_channels_heads
            ]
        )
        self.register_parameter(
            "sigma_eps",
            nn.Parameter(torch.tensor([self.sigma_eps_default]), requires_grad=False),
        )
        if init_custom:
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            detection_conv = next(m for m in self.mt_heads[0].core if isinstance(m, nn.Conv2d))
            nn.init.kaiming_normal_(detection_conv.weight, mode="fan_in", nonlinearity="relu")
            nn.init.kaiming_normal_(self.mt_heads[0].out_conv.weight, mode="fan_in", nonlinearity="linear")
            nn.init.constant_(self.mt_heads[0].out_conv.bias, -6.0)

    def _forward_core(self, x: torch.Tensor) -> torch.Tensor:
        if self.ch_in == 3:
            o0 = self.unet_shared(x[:, [0]])
            o1 = self.unet_shared(x[:, [1]])
            o2 = self.unet_shared(x[:, [2]])
            o = torch.cat((o0, o1, o2), dim=1)
        else:
            o = self.unet_shared(x)
        return self.unet_union(o)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._forward_core(x)
        raw = torch.cat([head(h) for head in self.mt_heads], dim=1)
        p = torch.sigmoid(torch.clamp(raw[:, 0:1], min=-8.0, max=8.0))
        phot = torch.sigmoid(raw[:, 1:2])
        xyz = torch.tanh(raw[:, 2:5])
        sig = torch.sigmoid(raw[:, 5:9]) * 3.0 + self.sigma_eps
        bg = torch.sigmoid(raw[:, 9:10])
        return torch.cat([p, phot, xyz, sig, bg], dim=1)


def model_from_params(param):
    hp = param["HyperParameter"]
    ap = hp["arch_param"]
    act = _get_activation(ap.get("activation", "ReLU"))
    return SigmaMUNet(
        ch_in=int(hp["channels_in"]),
        depth_shared=int(ap["depth_shared"]),
        depth_union=int(ap["depth_union"]),
        initial_features=int(ap["initial_features"]),
        inter_features=int(ap["inter_features"]),
        norm=ap.get("norm"),
        norm_groups=ap.get("norm_groups"),
        norm_head=ap.get("norm_head"),
        norm_head_groups=ap.get("norm_head_groups"),
        pool_mode=ap.get("pool_mode", "StrideConv"),
        upsample_mode=ap.get("upsample_mode", "nearest"),
        skip_gn_level=ap.get("skip_gn_level"),
        activation=act,
        init_custom=ap.get("init_custom", True),
    )


def load_sigma_munet(param_yaml: Union[str, Path], weights_pt: Union[str, Path], device: str = "cpu"):
    """Load trained model_best.pt + param_run_used.yaml in pure PyTorch with strict=True."""
    param = yaml.safe_load(Path(param_yaml).read_text())
    model = model_from_params(param).to(device)
    try:
        state = torch.load(str(weights_pt), map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(str(weights_pt), map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, param


def resolve_infer_device(device_arg: str) -> str:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device_arg
