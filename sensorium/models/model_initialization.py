import torch
from torch import nn
from nnfabrik.builder import get_model



def init_model(model_name, checkpoint_path, dataloaders):
    if model_name == 'generalization':
        model = sota(dataloaders, checkpoint_path)
    elif model_name == 'ensemble':
        checkpoints = [f'{checkpoint_path}_{n}.pth' for n in range(41, 51)]
        model = SotaEnsemble(dataloaders, checkpoints).cuda()
    else:
        raise ValueError(f'Unknown model {model_name}')
    return model


class SotaEnsemble(nn.Module):
    def __init__(self, dataloaders, checkpoint_paths):
        super().__init__()
        self.models = []
        # We need to register dull param
        # to make neurapredictors functions correctly assign the device
        self.param = nn.Parameter(torch.tensor([42.]))

        for path in checkpoint_paths:
            self.models.append(sota(dataloaders, path).cuda())

    def __call__(self, *args, **kwargs):
        return torch.mean(torch.stack([
            model(*args, **kwargs) for model in self.models
        ]), dim=0)



def sota(dataloaders, checkpoint_path):
    model_fn = 'sensorium.models.stacked_core_full_gauss_readout'
    model_config = {
        'pad_input': False,
        'layers': 4,
        'input_kern': 9,
        'gamma_input': 6.3831,
        'gamma_readout': 0.0076,
        'hidden_kern': 7,
        'hidden_channels': 64,
        'depth_separable': True,
        'grid_mean_predictor': {
            'type': 'cortex',
            'input_dimensions': 2,
            'hidden_layers': 1,
            'hidden_features': 30,
            'final_tanh': True
        },
        'init_sigma': 0.1,
        'init_mu_range': 0.3,
        'gauss_type': 'full',
        'shifter': False,
        'stack': -1,
    }

    model = get_model(model_fn=model_fn,
                      model_config=model_config,
                      dataloaders=dataloaders,
                      seed=42,)

    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model


def ln_model(dataloaders, checkpoint_path):
    model_fn = 'sensorium.models.stacked_core_full_gauss_readout'
    model_config = {
        'pad_input': False,
        'stack': -1,
        'layers': 3,
        'input_kern': 9,
        'gamma_input': 6.3831,
        'gamma_readout': 0.0076,
        'hidden_kern': 7,
        'hidden_channels': 64,
        'grid_mean_predictor': {
            'type': 'cortex',
            'input_dimensions': 2,
            'hidden_layers': 1,
            'hidden_features': 30,
            'final_tanh': True
        },
        'depth_separable': True,
        'init_sigma': 0.1,
        'init_mu_range': 0.3,
        'gauss_type': 'full',
        'linear': True
    }
    model = get_model(model_fn=model_fn,
                      model_config=model_config,
                      dataloaders=dataloaders,
                      seed=42, )
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model
