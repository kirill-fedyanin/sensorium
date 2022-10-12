from PIL import Image
import requests
import matplotlib.pyplot as plt

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T

# torch.set_grad_enabled(False)


class SensorDEMO(nn.Module):
    def __init__(
            self, hidden_dim=256, nheads=8,
            num_encoder_layers=6, num_decoder_layers=6, num_neurons=8372,
            last_hidden_dim=8, backbone=None, data_key=None
        ):
        super().__init__()
        self.data_key = data_key

        # create ResNet-50 backbone
        if backbone is None:
            self.backbone = resnet50()
            self.backbone.fc = nn.Sequential()
        else:
            self.backbone = backbone

        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(num_neurons, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(60, 32))
        self.col_embed = nn.Parameter(torch.rand(60, 32))

        self.linear = nn.Linear(64, 1)

    def forward(
            self,
            inputs,
            *args,
            targets=None,
            data_key=None,
            behavior=None,
            pupil_center=None,
            trial_idx=None,
            shift=None,
            detach_core=False,
            **kwargs
    ):
        """
        data_key just to follow sensorium trainer
        """
        # import ipdb; ipdb.set_trace()
        h = self.backbone(inputs)

        # construct positional encodings
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        # propagate through the transformer
        batch_size = h.shape[0]
        h = self.transformer(
            pos + 0.1 * h.flatten(2).permute(2, 0, 1), self.query_pos.unsqueeze(1).repeat(1, batch_size, 1)
        ).transpose(0, 1)
        h = self.linear(h).squeeze(-1)
        return h
