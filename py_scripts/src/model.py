"""Pre-activation ResNet for CIFAR-10 (32x32 inputs).

Design (default config, ~4.70M trainable parameters):
  stem   : 3x3 conv, 3 -> 64, stride 1 (no max-pool, keep 32x32 resolution)
  stage 1: 4 pre-activation basic blocks, 64 ch,  32x32
  stage 2: 4 pre-activation basic blocks, 128 ch, 16x16 (first block stride 2)
  stage 3: 3 pre-activation basic blocks, 256 ch, 8x8   (first block stride 2)
  head   : BN -> ReLU -> global average pooling -> FC(256 -> 10)

Residual block:  y = S(x) + F(x)
  F(x) = Conv3x3(ReLU(BN(Conv3x3(ReLU(BN(x))))))      (pre-activation, He et al. 2016b)
  S(x) = x                                             if shape is unchanged
       = Conv1x1(AvgPool2x2(x))                        if downsampling (ResNet-D shortcut, He et al. 2019)
       = Conv1x1(x)                                    if only the channel count changes

The same class also builds the WRN-28-10 teacher used for knowledge distillation
(widths 160/320/640, blocks 4/4/4, stem 16).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.dropout = dropout

        self.shortcut = None
        if stride != 1 or in_ch != out_ch:
            layers = [nn.AvgPool2d(stride, stride)] if stride != 1 else []
            layers.append(nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False))
            self.shortcut = nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        # projection shortcut takes the pre-activated input (as in PreAct-ResNet / WRN)
        sc = x if self.shortcut is None else self.shortcut(out)
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        if self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.conv2(out)
        return out + sc


class PreActResNet(nn.Module):
    def __init__(self, widths=(64, 128, 256), blocks=(4, 4, 3), stem_channels=64,
                 dropout=0.0, num_classes=10):
        super().__init__()
        assert len(widths) == len(blocks)
        self.stem = nn.Conv2d(3, stem_channels, 3, 1, 1, bias=False)

        layers, in_ch = [], stem_channels
        for stage, (out_ch, n) in enumerate(zip(widths, blocks)):
            for i in range(n):
                stride = 2 if (stage > 0 and i == 0) else 1
                layers.append(PreActBlock(in_ch, out_ch, stride, dropout))
                in_ch = out_ch
        self.layers = nn.Sequential(*layers)

        self.bn = nn.BatchNorm2d(in_ch)
        self.fc = nn.Linear(in_ch, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.layers(self.stem(x))
        out = F.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


def build_model(mcfg):
    return PreActResNet(
        widths=tuple(mcfg["widths"]),
        blocks=tuple(mcfg["blocks"]),
        stem_channels=mcfg.get("stem_channels", mcfg["widths"][0]),
        dropout=mcfg.get("dropout", 0.0),
        num_classes=mcfg.get("num_classes", 10),
    )


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
