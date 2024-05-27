import numpy as np
import tinycudann as tcnn
import torch
from .geometry_net import _MLP, NeRFEncoding, WindowedNeRFEncoding, MLP_CLASSES
import logging

LOGGER = logging.getLogger(__name__)
print = LOGGER.info

def get_non_rigid_offset_net(num_outputs: int, non_rigid_offset_input_dim: int, non_rigid_offset_net_encoding_type: str, non_rigid_offset_net_num_freq: int):
    if non_rigid_offset_net_encoding_type == 'hash_grid':
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
        encoding = tcnn.Encoding(3, config["encoding"])
        network = tcnn.Network(encoding.n_output_dims + non_rigid_offset_input_dim, num_outputs, config["network"])

    # elif non_rigid_offset_net_encoding_type == 'frequency':
    #     config = {
    #         "encoding": {
    #             "otype": "Composite",
    #             "nested": [
    #                 {
    #                     "otype": "Frequency",
    #                     "n_dims_to_encode": 3,
    #                     "n_frequencies": non_rigid_offset_net_num_freq,
    #                 },
    #                 {
    #                     "otype": "Identity"
    #                 }
    #             ]
    #         },
    #         "network": {
    #             "otype": "FullyFusedMLP",
    #             "activation": "ReLU",
    #             "output_activation": "None",
    #             "n_neurons": 128,
    #             "n_hidden_layers": 4,
    #         },
    #     }
    #     encoding = tcnn.Encoding(3, config["encoding"])
    #     network = tcnn.Network(encoding.n_output_dims + non_rigid_offset_input_dim, num_outputs, config["network"])

    elif non_rigid_offset_net_encoding_type in MLP_CLASSES:
        config = {
            "encoding": {
                "otype": "Composite",
                "nested": [
                    {
                        "otype": "Frequency",
                        "n_dims_to_encode": 3,
                        "n_frequencies": non_rigid_offset_net_num_freq,
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

        enconding_type = WindowedNeRFEncoding if "windowed" in non_rigid_offset_net_encoding_type else NeRFEncoding
        print(f"Non rigid offset net: Encoding type: {enconding_type.__name__}")
        encoding = enconding_type(in_dim=in_dim, num_frequencies=num_frequencies, min_freq_exp=0.0, max_freq_exp=num_frequencies - 1, include_input=True)

        use_lipschitz = 'lipschitz' in non_rigid_offset_net_encoding_type
        print(f"Non rigid offset net: Use lipschitz: {use_lipschitz}")

        network = _MLP(encoding.get_out_dim()+ non_rigid_offset_input_dim, num_outputs, config["network"], use_lipschitz=use_lipschitz)
        torch.nn.init.uniform_(network.net[-1].weight, a=-1e-5, b=1e-5)
        print(f"Non rigid offset net: network.net[-1], weight min max: {network.net[-1].weight.min().item(), network.net[-1].weight.max().item()} ")

    else:
        raise ValueError(f'Unknown model type: {non_rigid_offset_net_encoding_type}')

    # encoding = tcnn.Encoding(3, config["encoding"])
    # network = tcnn.Network(encoding.n_output_dims + non_rigid_offset_input_dim, num_outputs, config["network"])

    return NonRigidOffsetNet(encoding, network)


class NonRigidOffsetNet(torch.nn.Module):
    def __init__(self, encoding: torch.nn.Module, network: torch.nn.Module):
        super().__init__()
        self.encoding = encoding
        self.network = network

    def forward(self, x, params):
        batch_size = params.shape[0]
        params_dim = params.shape[-1]
        num_verts = x.shape[1]
        params = params.unsqueeze(1).expand(batch_size, num_verts, -1)  # repeat to every vertex in a batch
        x = x.expand(batch_size, num_verts, -1)  # repeat to every vertex in a batch

        x_encoded = self.encoding(x.reshape(-1, 3)) # (B, V, feature_dim)
        x_encoded = torch.cat([x_encoded, params.reshape(-1, params_dim)], dim=-1)

        return self.network(x_encoded).view(batch_size, num_verts, -1)

    def __getitem__(self, i):
        return self.encoding if i == 0 else self.network

    def __len__(self):
        return 2