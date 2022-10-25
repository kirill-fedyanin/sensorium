from PIL import Image
import requests
import matplotlib.pyplot as plt

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T

# torch.set_grad_enabled(False)


class FakeReadout:
    def __init__(self, base_model):
        self.base_model = base_model

    def regularizer(self, data_key=None):
        return self.base_model.regularizer(data_key)


class SensorDEMOdebug(nn.Module):
    def __init__(
            self, hidden_dim=256, nheads=8,
            num_encoder_layers=6, num_decoder_layers=6, num_neurons=8372,
            core=None, data_key=None
        ):
        super().__init__()
        self.data_key = data_key
        self.core = core
        self.linear = nn.Linear(6912, num_neurons)
        self.readout = FakeReadout(base_model=self)

    def regularizer(self, data_key=None):
        return 0

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
        h = self.core(inputs)
        h = h.view(h.shape[0], -1)
        return self.linear(h)


class SensorDEMO(nn.Module):
    def __init__(
            self, hidden_dim=256, nheads=8,
            num_encoder_layers=6, num_decoder_layers=6, num_neurons=8372,
            core=None, data_key=None
        ):
        super().__init__()
        self.data_key = data_key
        self.core = core
        self.conv = nn.Conv2d(hidden_dim, hidden_dim // 2, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim // 2, nheads, num_encoder_layers, num_decoder_layers
        )

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(num_neurons, hidden_dim // 2))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(60, 16))
        self.col_embed = nn.Parameter(torch.rand(60, 16))

        self.norm = nn.LayerNorm(32)
        self.linear = nn.Linear(32, 1)
        self.readout = FakeReadout(base_model=self)

    def regularizer(self, data_key=None):
        return 0
        # import ipdb; ipdb.set_trace()
        # pass

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
        h = self.core(inputs)
        h = self.conv(h)
        # # construct positional encodings
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)
        #
        # # propagate through the transformer
        batch_size = h.shape[0]
        h = self.transformer(
            pos + 0.1 * h.flatten(2).permute(2, 0, 1), self.query_pos.unsqueeze(1).repeat(1, batch_size, 1)
        ).transpose(0, 1)
        h = self.norm(h)
        h = self.linear(h).squeeze(-1)
        return nn.functional.elu(h) + 1
