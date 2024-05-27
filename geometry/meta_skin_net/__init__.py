
import logging
import torch
from collections import OrderedDict
import numpy as np
from .decoder import Deformer

LOGGER = logging.getLogger(__name__)
print = LOGGER.info

def get_meta_skinning_weight_net(weight_path=None):
    ''' Returns Skinning Model instances.
    Args:
        cfg (yaml config): yaml config object
        dim (int): points dimension
        init_weights (bool): whether to initialize the weights for the skinning network with pre-trained model (MetaAvatar)

    Deformer(
    (lin0): Linear(in_features=3, out_features=128, bias=True)
    (lin1): Linear(in_features=128, out_features=128, bias=True)
    (lin2): Linear(in_features=128, out_features=128, bias=True)
    (lin3): Linear(in_features=128, out_features=128, bias=True)
    (lin4): Linear(in_features=128, out_features=25, bias=True)
    (activation): Softplus(beta=100, threshold=20)

    odict_keys(['decoder.net.net.0.0.weight', 'decoder.net.net.0.0.bias', 'decoder.net.net.1.0.weight', 'decoder.net.net.1.0.bias', 'decoder.net.net.2.0.weight', 'decoder.net.net.2.0.bias', 'decoder.net.net.3.0.weight', 'decoder.net.net.3.0.bias', 'decoder.net.net.4.0.weight', 'decoder.net.net.4.0.bias', 'decoder.net.net.5.0.weight', 'decoder.net.net.5.0.bias', 'decoder.net.net.6.0.weight', 'decoder.net.net.6.0.bias', 
    'skinning_decoder_fwd.lin0.bias', 'skinning_decoder_fwd.lin0.weight_g', 'skinning_decoder_fwd.lin0.weight_v', 'skinning_decoder_fwd.lin1.bias', 'skinning_decoder_fwd.lin1.weight_g', 'skinning_decoder_fwd.lin1.weight_v', 'skinning_decoder_fwd.lin2.bias', 'skinning_decoder_fwd.lin2.weight_g', 'skinning_decoder_fwd.lin2.weight_v', 'skinning_decoder_fwd.lin3.bias', 'skinning_decoder_fwd.lin3.weight_g', 'skinning_decoder_fwd.lin3.weight_v', 'skinning_decoder_fwd.lin4.bias', 'skinning_decoder_fwd.lin4.weight_g', 'skinning_decoder_fwd.lin4.weight_v', 
    'skinning_decoder_bwd.lin0.bias', 'skinning_decoder_bwd.lin0.weight_g', 'skinning_decoder_bwd.lin0.weight_v', 'skinning_decoder_bwd.lin1.bias', 'skinning_decoder_bwd.lin1.weight_g', 'skinning_decoder_bwd.lin1.weight_v', 'skinning_decoder_bwd.lin2.bias', 'skinning_decoder_bwd.lin2.weight_g', 'skinning_decoder_bwd.lin2.weight_v', 'skinning_decoder_bwd.lin3.bias', 'skinning_decoder_bwd.lin3.weight_g', 'skinning_decoder_bwd.lin3.weight_v', 'skinning_decoder_bwd.lin4.bias', 'skinning_decoder_bwd.lin4.weight_g', 'skinning_decoder_bwd.lin4.weight_v'])
    '''

    snarf_cfg = {'d_in': 3, 'd_out': 25, 'd_hidden': 128, 'n_layers': 4, 'skip_in': [], 'cond_in': [], 'multires': 0, 'bias': 1.0, 'geometric_init': False, 'weight_norm': True}
    decoder = Deformer(**snarf_cfg)


    if weight_path is not None:
        print(f"loading skinning weights from {weight_path}")
        ckpt = torch.load(weight_path, map_location='cpu') # dict_keys(['epoch_it', 'it', 'loss_val_best', 'model', 'optimizer'])

        skinning_decoder_fwd_state_dict = OrderedDict()
        for k, v in ckpt['model'].items():
            if k.startswith('module'):
                k = k[7:]
            if k.startswith('skinning_decoder_fwd'):
                skinning_decoder_fwd_state_dict[k[21:]] = v

        decoder.load_state_dict(skinning_decoder_fwd_state_dict, strict=False)

    return decoder

def _softmax(x):
    return torch.nn.functional.softmax(x, dim=-1)

def _sigmoid(x):
    return torch.sigmoid(x)

''' Hierarchical softmax following the kinematic tree of the human body. Imporves convergence speed'''
def hierarchical_softmax(x):

    *spatial_shape, feature_dim = x.shape
    spatial_shape_product = np.product(spatial_shape)

    x = x.view(-1, feature_dim)

    prob_all = torch.ones(spatial_shape_product, 24, device=x.device)

    prob_all[:, [1, 2, 3]] = prob_all[:, [0]] * _sigmoid(x[:, [0]]) * _softmax(x[:, [1, 2, 3]])
    prob_all[:, [0]] = prob_all[:, [0]] * (1 - _sigmoid(x[:, [0]]))

    prob_all[:, [4, 5, 6]] = prob_all[:, [1, 2, 3]] * (_sigmoid(x[:, [4, 5, 6]]))
    prob_all[:, [1, 2, 3]] = prob_all[:, [1, 2, 3]] * (1 - _sigmoid(x[:, [4, 5, 6]]))

    prob_all[:, [7, 8, 9]] = prob_all[:, [4, 5, 6]] * (_sigmoid(x[:, [7, 8, 9]]))
    prob_all[:, [4, 5, 6]] = prob_all[:, [4, 5, 6]] * (1 - _sigmoid(x[:, [7, 8, 9]]))

    prob_all[:, [10, 11]] = prob_all[:, [7, 8]] * (_sigmoid(x[:, [10, 11]]))
    prob_all[:, [7, 8]] = prob_all[:, [7, 8]] * (1 - _sigmoid(x[:, [10, 11]]))

    prob_all[:, [12, 13, 14]] = prob_all[:, [9]] * _sigmoid(x[:, [24]]) * _softmax(x[:, [12, 13, 14]])
    prob_all[:, [9]] = prob_all[:, [9]] * (1 - _sigmoid(x[:, [24]]))

    prob_all[:, [15]] = prob_all[:, [12]] * (_sigmoid(x[:, [15]]))
    prob_all[:, [12]] = prob_all[:, [12]] * (1 - _sigmoid(x[:, [15]]))

    prob_all[:, [16, 17]] = prob_all[:, [13, 14]] * (_sigmoid(x[:, [16, 17]]))
    prob_all[:, [13, 14]] = prob_all[:, [13, 14]] * (1 - _sigmoid(x[:, [16, 17]]))

    prob_all[:, [18, 19]] = prob_all[:, [16, 17]] * (_sigmoid(x[:, [18, 19]]))
    prob_all[:, [16, 17]] = prob_all[:, [16, 17]] * (1 - _sigmoid(x[:, [18, 19]]))

    prob_all[:, [20, 21]] = prob_all[:, [18, 19]] * (_sigmoid(x[:, [20, 21]]))
    prob_all[:, [18, 19]] = prob_all[:, [18, 19]] * (1 - _sigmoid(x[:, [20, 21]]))

    prob_all[:, [22, 23]] = prob_all[:, [20, 21]] * (_sigmoid(x[:, [22, 23]]))
    prob_all[:, [20, 21]] = prob_all[:, [20, 21]] * (1 - _sigmoid(x[:, [22, 23]]))

    prob_all = prob_all.reshape(*spatial_shape, prob_all.shape[-1])
    return prob_all