import torchvision.models
from torch import nn
from torchvision.models.feature_extraction import create_feature_extractor


class RegNetX_3_2GF(nn.Module):
    def __init__(self, output_stride):
        super().__init__()
        self.output_node = 'trunk_output.block4.block4-1.activation'
        trunk = getattr(torchvision.models, 'regnet_x_3_2gf')(pretrained=True)

        s = 2
        for n, m in trunk.named_modules():
            if not isinstance(m, nn.Conv2d):
                continue

            if m.stride == (2, 2) and m.kernel_size == (1, 1):
                s *= 2

            if s > output_stride:
                m.stride = 1
                if m.kernel_size == (3, 3):
                    rate = int(s / output_stride)
                    m.dilation = m.padding = rate

        self.trunk = create_feature_extractor(trunk, return_nodes=[self.output_node])

    def forward(self, images):
        return self.trunk(images)[self.output_node]


class _ResNet(nn.Module):
    def __init__(self, depth, output_stride, multi_grid, output_node='layer4.2.relu_2'):
        super().__init__()
        self.output_node = output_node

        trunk = getattr(torchvision.models, 'resnet' + str(depth))(
            pretrained=True,
            replace_stride_with_dilation=[output_stride < (2 ** i) for i in range(3, 6)],
        )

        i = 0
        for n, m in trunk.named_modules():
            if not isinstance(m, nn.Conv2d):
                continue
            if n.startswith('layer4') and m.kernel_size == (3, 3):
                m.padding = m.dilation = int(multi_grid[i] * 32 / output_stride)
                i += 1

        self.trunk = create_feature_extractor(trunk, return_nodes=[self.output_node])

    def forward(self, images):
        return self.trunk(images)[self.output_node]


class ResNet50(_ResNet):
    def __init__(self, output_stride, multi_grid=[1, 2, 4]):
        super().__init__(50, output_stride, multi_grid)


class ResNet101(_ResNet):
    def __init__(self, output_stride, multi_grid=[1, 2, 4]):
        super().__init__(101, output_stride, multi_grid)


class MobileNetV2(nn.Module):
    def __init__(self, output_stride):
        super().__init__()
        self.output_node = 'features.17.conv.3'

        s = 1
        trunk = torchvision.models.mobilenet_v2(pretrained=True)
        for m in trunk.modules():
            if not isinstance(m, nn.Conv2d):
                continue

            if m.stride == (2, 2):
                s *= 2
                if s > output_stride:
                    m.stride = 1
                    continue

            if (s > output_stride) and (m.kernel_size != (1, 1)):
                rate = int(s / output_stride)
                m.dilation = m.padding = rate

        self.trunk = create_feature_extractor(trunk, return_nodes=[self.output_node])

    def forward(self, images):
        return self.trunk(images)[self.output_node]
