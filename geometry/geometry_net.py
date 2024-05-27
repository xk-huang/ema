import numpy as np
import tinycudann as tcnn
import torch
import logging

LOGGER = logging.getLogger(__name__)
print = LOGGER.info

MLP_CLASSES = ['frequency_windowed_lipschitz', 'frequency_lipschitz_windowed', 'frequency', 'frequency_lipschitz', 'frequency_windowed']

def get_geometry_net(learn_tet_vert_deform_with_mlp: bool, sdf_mlp_type: str, sdf_mlp_num_freq: int):
    if not isinstance(learn_tet_vert_deform_with_mlp, bool):
        raise ValueError('learn_tet_vert_deform_with_mlp must be a boolean')

    if sdf_mlp_type == 'hash_grid':
        num_levels=16
        features_per_level=2
        log2_hashmap_size=19
        base_res=16
        max_res=4096
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
            n_output_dims=1 + int(learn_tet_vert_deform_with_mlp) * 3,
            encoding_config=config["encoding"],
            network_config=config["network"],
        )
        # encoding = tcnn.Encoding(3, config["encoding"])
        # network = tcnn.Network(encoding.n_output_dims, 1 + int(learn_tet_vert_deform_with_mlp) * , config["network"])
        # network = _MLP(encoding.n_output_dims, 1 + int(learn_tet_vert_deform_with_mlp) * 3, config["network"])
        # model = torch.nn.Sequential(encoding, network)

    elif sdf_mlp_type in MLP_CLASSES:
        config = {
            "encoding": {
                "otype": "Composite",
                "nested": [
                    {
                        "otype": "Frequency",
                        "n_dims_to_encode": 3,
                        "n_frequencies": sdf_mlp_num_freq,
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
                "n_neurons": 256,
                "n_hidden_layers": 8,
            },
        }
        num_frequencies = config["encoding"]["nested"][0]["n_frequencies"]
        in_dim = config["encoding"]["nested"][0]["n_dims_to_encode"]

        enconding_type = WindowedNeRFEncoding if "windowed" in sdf_mlp_type else NeRFEncoding
        print(f"Geometry net: Encoding type: {enconding_type.__name__}")
        encoding = enconding_type(in_dim=in_dim, num_frequencies=num_frequencies, min_freq_exp=0.0, max_freq_exp=num_frequencies - 1, include_input=True)

        use_lipschitz = 'lipschitz' in sdf_mlp_type
        print(f"Geometry net: using {'Lipschitz' if use_lipschitz else 'Standard'} MLP")
        network = _MLP(encoding.get_out_dim(), 1 + int(learn_tet_vert_deform_with_mlp) * 3, config["network"], use_lipschitz=use_lipschitz)

        model = torch.nn.Sequential(encoding, network)
        # model = tcnn.NetworkWithInputEncoding(
        #     n_input_dims=3,
        #     n_output_dims=1 + int(learn_tet_vert_deform_with_mlp) * 3,
        #     encoding_config=config["encoding"],
        #     network_config=config["network"],
        # )
    else:
        raise ValueError(f'Unknown model type: {sdf_mlp_type}')

    return model

class _MLP(torch.nn.Module):
    def __init__(self, n_input_dims, n_output_dims, cfg, use_lipschitz=True):
        super(_MLP, self).__init__()
        self.use_lipschitz = use_lipschitz
        if self.use_lipschitz is True:
            self.linear_module = LipschitzLinear
        else:
            self.linear_module = torch.nn.Linear
        cfg['n_input_dims'] = n_input_dims
        cfg['n_output_dims'] = n_output_dims
        net = (self.linear_module(cfg['n_input_dims'], cfg['n_neurons'], bias=False), torch.nn.ReLU())
        for i in range(cfg['n_hidden_layers']-1):
            net = net + (self.linear_module(cfg['n_neurons'], cfg['n_neurons'], bias=False), torch.nn.ReLU())
        net = net + (self.linear_module(cfg['n_neurons'], cfg['n_output_dims'], bias=False),)
        self.net = torch.nn.Sequential(*net).cuda()

        self.net.apply(self._init_weights)

    def forward(self, x):
        return self.net(x.to(torch.float32))

    @staticmethod
    def _init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0.0)

    def get_lipschitz_loss(self):
        loss_lipschitz = 1.0
        for i in self.net.modules():
            if isinstance(i, LipschitzLinear):
                loss_lipschitz = loss_lipschitz * i.get_softplus_bound_c()

        return loss_lipschitz

class LipschitzLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super(LipschitzLinear, self).__init__(in_features, out_features, bias, device, dtype)
        self.bound_c = torch.nn.Parameter(torch.max(torch.sum(torch.abs(self.weight.detach()), dim=1)))
        ones = torch.ones(1, device=self.weight.device, dtype=self.weight.dtype)
        self.register_buffer('ones', ones, persistent=False)

    def forward(self, x):
        softplus_bound_c = self.get_softplus_bound_c()
        weight = self.weight_norm(self.weight, softplus_bound_c)
        return torch.nn.functional.linear(x, weight, self.bias)

    def weight_norm(self, weight, softplus_bound_c):
        absrowsum = torch.sum(torch.abs(weight), dim=1)
        scale = torch.minimum(self.ones, softplus_bound_c / absrowsum)
        return weight * scale.unsqueeze(-1)

    def get_softplus_bound_c(self):
        return torch.nn.functional.softplus(self.bound_c)

class NeRFEncoding(torch.nn.Module):
    """Multi-scale sinousoidal encodings. Support ``integrated positional encodings`` if covariances are provided.
    Each axis is encoded with frequencies ranging from 2^min_freq_exp to 2^max_freq_exp.
    Args:
        in_dim: Input dimension of tensor
        num_frequencies: Number of encoded frequencies per axis
        min_freq_exp: Minimum frequency exponent
        max_freq_exp: Maximum frequency exponent
        include_input: Append the input coordinate to the encoding
    """

    def __init__(
        self, in_dim: int, num_frequencies: int, min_freq_exp: float, max_freq_exp: float, include_input: bool = False
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = None
        self.num_frequencies = num_frequencies
        self.min_freq = min_freq_exp
        self.max_freq = max_freq_exp
        self.include_input = include_input
        print(f"NeRFEncoding: in_dim={in_dim}, num_frequencies={num_frequencies}, min_freq={min_freq_exp}, max_freq={max_freq_exp}, include_input={include_input}")

    def get_out_dim(self) -> int:
        if self.in_dim is None:
            raise ValueError("Input dimension has not been set")
        out_dim = self.in_dim * self.num_frequencies * 2
        if self.include_input:
            out_dim += self.in_dim
        return out_dim

    def forward(
        self,
        in_tensor, # : TensorType["bs":..., "input_dim"],
        covs=None, #: Optional[TensorType["bs":..., "input_dim", "input_dim"]] = None,
        **kwargs,
    ): # -> TensorType["bs":..., "output_dim"]:
        """Calculates NeRF encoding. If covariances are provided the encodings will be integrated as proposed
            in mip-NeRF.
        Args:
            in_tensor: For best performance, the input tensor should be between 0 and 1.
            covs: Covariances of input points.
        Returns:
            Output values will be between -1 and 1
        """
        original_in_tensor = (in_tensor - 0.5) * 2
        in_tensor = 2 * torch.pi * in_tensor  # scale to [0, 2pi]
        freqs = 2 ** torch.linspace(self.min_freq, self.max_freq, self.num_frequencies).to(in_tensor.device)
        scaled_inputs = in_tensor[..., None] * freqs  # [..., "input_dim", "num_scales"]
        scaled_inputs = scaled_inputs.view(*scaled_inputs.shape[:-2], -1)  # [..., "input_dim" * "num_scales"]

        if covs is None:
            encoded_inputs = torch.sin(torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1))
        else:
            input_var = torch.diagonal(covs, dim1=-2, dim2=-1)[..., :, None] * freqs[None, :] ** 2
            input_var = input_var.reshape((*input_var.shape[:-2], -1))
            encoded_inputs = expected_sin(
                torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1), torch.cat(2 * [input_var], dim=-1)
            )

        if self.include_input:
            encoded_inputs = torch.cat([encoded_inputs, original_in_tensor], dim=-1)
        return encoded_inputs

def expected_sin(x_means: torch.Tensor, x_vars: torch.Tensor) -> torch.Tensor:
    """Computes the expected value of sin(y) where y ~ N(x_means, x_vars)
    Args:
        x_means: Mean values.
        x_vars: Variance of values.
    Returns:
        torch.Tensor: The expected value of sin.
    """

    return torch.exp(-0.5 * x_vars) * torch.sin(x_means)

class WindowedNeRFEncoding(NeRFEncoding):
    def __init__(self, in_dim: int, num_frequencies: int, min_freq_exp: float, max_freq_exp: float, include_input: bool = False) -> None:
        super().__init__(in_dim, num_frequencies, min_freq_exp, max_freq_exp, include_input)
        # self.register_buffer("alpha", torch.tensor(self.max_freq + 1.0))
        self.register_buffer("alpha", torch.tensor(0.0))
        self.register_buffer("ones", torch.tensor(1.0))
        self.register_buffer("num_base_frequencies", torch.tensor(0.0))

    def forward(self, in_tensor, covs=None):
        features = super().forward(in_tensor, covs)
        alpha = self.alpha

        if self.include_input:
            features, identity = torch.split(features, (self.in_dim * self.num_frequencies * 2, self.in_dim), dim=-1)

        features = features.reshape(list(in_tensor.shape[:-1] + (self.in_dim, self.num_frequencies, 2)))

        window = self.cosine_easing_window(alpha, device=features.device).reshape(1, self.num_frequencies, 1)
        features = window * features
        features = features.reshape(list(in_tensor.shape[:-1]) + [-1])

        if self.include_input:
            return torch.cat([features, identity], dim=-1)
        else:
            return features
        
    def cosine_easing_window(self, alpha, device=None):
        # [NOTE] if band[i] <= 0, then i-th frequency is used.
        bands = torch.arange(self.num_frequencies, device=device) - self.num_base_frequencies
        # [NOTE] at the beginning, the alpha = 0.0, then gradually grows.
        x = torch.clamp(alpha - bands, 0.0, 1.0)
        return 0.5 * (1 + torch.cos(torch.pi * x + torch.pi))

    def register_alpha(self, alpha):
        self.alpha = self.ones * alpha

    def register_num_base_frequencies(self, num_base_frequencies):
        if num_base_frequencies < 0:
            LOGGER.warning(f"num_base_frequencies should be positive. Got {num_base_frequencies}."
            "Then there are `postional_encoding_anneal_steps / num_freq * -num_base_frequencies`"
            "steps where no frequencies are used.")
        print(f"Registering num_base_frequencies: {num_base_frequencies}")
        self.num_base_frequencies = self.ones * num_base_frequencies