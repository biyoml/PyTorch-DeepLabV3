import torch
from torch import nn


class _ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1, dilation=1):
        super().__init__(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=kernel_size,
                      dilation=dilation,
                      padding=int((kernel_size - 1) * dilation / 2),
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class _ASPP(nn.Module):
    def __init__(self, feature_shape, out_channels, rates, image_pooling):
        super().__init__()
        branches = []
        branches.append(_ConvBNReLU(feature_shape[0], out_channels))   # 1x1 conv

        if rates is not None:
            for r in rates:
                branches.append(_ConvBNReLU(feature_shape[0], out_channels, 3, r))

        if image_pooling:
            branches.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    _ConvBNReLU(feature_shape[0], out_channels),
                    nn.Upsample(feature_shape[1:],
                                mode='bilinear',
                                align_corners=True)
                )
            )

        self.branches = nn.ModuleList(branches)
        self.proj = _ConvBNReLU(
            out_channels * len(branches),
            out_channels
        )

    def forward(self, x):
        x = [b(x) for b in self.branches]
        x = torch.cat(x, axis=1)
        return self.proj(x)


def _get_feature_shape(model, input_size):
    inp = torch.randn([2, 3, input_size, input_size])
    with torch.no_grad():
        out = model(inp)
    return out.shape[1:]


class DeepLabV3(nn.Sequential):
    def __init__(self, backbone, input_size, num_classes, aspp_rates,
                 image_pooling, dropout_rate):
        feature_shape = _get_feature_shape(backbone, input_size)
        aspp = _ASPP(feature_shape, 256, aspp_rates, image_pooling)
        head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, num_classes, kernel_size=1),
            nn.Upsample(input_size,
                        mode='bilinear',
                        align_corners=True)
        )
        super().__init__(backbone, aspp, head)
