import numpy as np
import tinycudann as tcnn
import torch
from .geometry_net import _MLP, NeRFEncoding, WindowedNeRFEncoding, MLP_CLASSES
import logging

LOGGER = logging.getLogger(__name__)
print = LOGGER.info

def get_skinning_weight_net(num_tfs: int, skin_net_encoding_type: str, skin_net_num_freq: int):
    if skin_net_encoding_type == 'hash_grid':
        num_levels=16
        features_per_level=2
        log2_hashmap_size=19
        base_res=16
        max_res=2048
        growth_factor = np.exp((np.log(max_res) - np.log(base_res)) / (num_levels - 1))
        
        num_layers: int = 3
        hidden_dim: int = 32

        config = {
            "encoding": {
                "otype": "HashGrid",
                "n_levels": num_levels,
                "n_features_per_level": features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_res,
                "per_level_scale": growth_factor,
            },
            "network": {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            },
        }
        model = tcnn.NetworkWithInputEncoding(
            n_input_dims=3,
            n_output_dims=num_tfs,
            encoding_config=config["encoding"],
            network_config=config["network"],
        )
    elif skin_net_encoding_type in MLP_CLASSES:
        config = {
            "encoding": {
                "otype": "Composite",
                "nested": [
                    {
                        "otype": "Frequency",
                        "n_dims_to_encode": 3,
                        "n_frequencies": skin_net_num_freq,
                    },
                    {
                        "otype": "Identity"
                    }
                ]
            },
            "network": {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 128,
                "n_hidden_layers": 4,
            },
        }
        num_frequencies = config["encoding"]["nested"][0]["n_frequencies"]
        in_dim = config["encoding"]["nested"][0]["n_dims_to_encode"]

        enconding_type = WindowedNeRFEncoding if "windowed" in skin_net_encoding_type else NeRFEncoding
        print(f"Skin net: Encoding type: {enconding_type.__name__}")
        encoding = enconding_type(in_dim=in_dim, num_frequencies=num_frequencies, min_freq_exp=0.0, max_freq_exp=num_frequencies - 1, include_input=True)

        use_lipschitz = 'lipschitz' in skin_net_encoding_type
        print(f"Skin net: Use lipschitz: {use_lipschitz}")
        network = _MLP(encoding.get_out_dim(), num_tfs, config["network"], use_lipschitz=use_lipschitz)

        model = torch.nn.Sequential(encoding, network)
    else:
        raise ValueError(f'Unknown model type: {skin_net_encoding_type}')


    # encoding = tcnn.Encoding(3, config["encoding"])
    # network = tcnn.Network(encoding.n_output_dims, num_tfs, config["network"])
    # [NOTE] cannot use culass, internal error. ~~change to mlp according to https://github.com/NVlabs/nvdiffrec/issues/5/~~
    # ~~maybe~~ it's because of the fail of dmtet, num vertices = 0
    # the real problem cames from fp32->fp16->fp32->fp16->..., resulting in nan in backward, then leads to nan sdf value, and finally empty mesh

    # network = _MLP(encoding.n_output_dims, num_tfs, config["network"])
    # model = torch.nn.Sequential(encoding, network)
    return model
