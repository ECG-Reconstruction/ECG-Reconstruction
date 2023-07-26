import torch
from torch import nn


module = nn.Sequential(
    nn.Conv1d(
        in_channels=224,
        out_channels=448,
        kernel_size=9,
        padding="same",
        dilation=9604,
    ),
    nn.Conv1d(
        in_channels=448,
        out_channels=448,
        kernel_size=9,
        padding="same",
        dilation=19208,
    ),
).cuda()
module(torch.rand(1, 224, 5000).cuda()).mean().backward()
