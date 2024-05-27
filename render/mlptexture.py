# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import logging
import torch
import tinycudann as tcnn
import numpy as np

LOGGER = logging.getLogger(__name__)

#######################################################################################################################################################
# Small MLP using PyTorch primitives, internal helper class
#######################################################################################################################################################

class _MLP(torch.nn.Module):
    def __init__(self, cfg, loss_scale=1.0):
        super(_MLP, self).__init__()
        self.loss_scale = loss_scale
        backbone = (torch.nn.Linear(cfg['n_input_dims'], cfg['n_neurons'], bias=False), torch.nn.ReLU())
        for i in range(cfg['n_hidden_layers']-1):
            backbone = backbone + (torch.nn.Linear(cfg['n_neurons'], cfg['n_neurons'], bias=False), torch.nn.ReLU())
        self.backbone = torch.nn.Sequential(*backbone).cuda()
        if cfg['n_output_dims'] > 0:
            self.output_layer = torch.nn.Linear(cfg['n_neurons'], cfg['n_output_dims'], bias=False).cuda()
        else:
            self.output_layer = None

        if cfg["n_cond_input_dims"] > 0:
            conditonal_sub_net = [torch.nn.Linear(cfg['n_neurons'] + cfg["n_cond_input_dims"], cfg['n_neurons'], bias=False)]
            conditonal_sub_net += [torch.nn.Linear(cfg['n_neurons'], max(1, cfg['n_neurons'] // 2), bias=False), torch.nn.ReLU()]
            # self.conditonal_sub_net_projector = torch.nn.Linear(cfg['n_neurons'], cfg['n_neurons'], bias=False).cuda()
            # conditonal_sub_net = [torch.nn.Linear(cfg['n_neurons'] + cfg["n_cond_input_dims"], max(1, cfg['n_neurons'] // 2), bias=False), torch.nn.ReLU()]
            conditonal_sub_net += [torch.nn.Linear(max(1, cfg['n_neurons'] // 2), cfg['n_cond_output_dims'], bias=False)]
            self.conditonal_sub_net = torch.nn.Sequential(*conditonal_sub_net).cuda()
        else:
            self.conditonal_sub_net = None
        
        self.backbone.apply(self._init_weights)
        if self.output_layer is not None:
            self.output_layer.apply(self._init_weights)
        if self.conditonal_sub_net:
            self.conditonal_sub_net.apply(self._init_weights)
        
        if self.loss_scale != 1.0:
            self.backbone.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] * self.loss_scale, ))
            if self.output_layer is not None:
                self.output_layer.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] * self.loss_scale, ))
            if self.conditonal_sub_net:
                self.conditonal_sub_net.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] * self.loss_scale, ))


    def forward(self, x, conditonal_input=None):
        # x = torch.cat([x, conditonal_input], dim=-1)
        interm_x = self.backbone(x)
        uncond_x, cond_x = None, None
        if self.output_layer is not None:
            uncond_x = self.output_layer(interm_x)
        if conditonal_input is not None and self.conditonal_sub_net is not None:
            cond_x = self.conditonal_sub_net(torch.cat([interm_x, conditonal_input], dim=-1))
            # cond_x = self.conditonal_sub_net(torch.cat([self.conditonal_sub_net_projector(interm_x), conditonal_input], dim=-1))
        return uncond_x, cond_x

    @staticmethod
    def _init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0.0)

#######################################################################################################################################################
# Outward visible MLP class
#######################################################################################################################################################

class MLPTexture3D(torch.nn.Module):
    def __init__(self, AABB, channels = 3, internal_dims = 32, hidden = 2, min_max = None, FLAGS = None):
        super(MLPTexture3D, self).__init__()

        if FLAGS is None or FLAGS.local_rank == 0:
            print = LOGGER.info
        else:
            print = LOGGER.debug

        self.channels = channels
        self.internal_dims = internal_dims
        self.AABB = AABB
        self.min_max = min_max
        print("MLPTexture3D: channels = {}, internal_dims = {}, AABB = {}, min_max = {}".format(channels, internal_dims, AABB, min_max))

        # Setup positional encoding, see https://github.com/NVlabs/tiny-cuda-nn for details
        desired_resolution = 4096
        base_grid_resolution = 16
        num_levels = 16
        per_level_scale = np.exp(np.log(desired_resolution / base_grid_resolution) / (num_levels-1))

        enc_cfg =  {
            "otype": "HashGrid",
            "n_levels": num_levels,
            "n_features_per_level": 2,
            "log2_hashmap_size": 19,
            "base_resolution": base_grid_resolution,
            "per_level_scale" : per_level_scale
	    }

        # gradient_scaling = 128.0  # shape texture: https://github.com/NVlabs/nvdiffrec/issues/5
        gradient_scaling = 1.0  # [NOTE] Otherwise FAIL to generate good texture
        self.encoder = tcnn.Encoding(3, enc_cfg)
        self.encoder.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] / gradient_scaling, ))

        self.use_texture_conditional_inputs = FLAGS.use_texture_conditional_inputs
        if self.use_texture_conditional_inputs is True:
            # set conditional inputs dim
            self.texture_conditional_input_dims = FLAGS.texture_conditional_input_dims
            print(f"use condtional inputs = True; texture_conditional_input_dims = {self.texture_conditional_input_dims}")

            # only normal use conditional inputs
            if FLAGS.texture_conditional_channel_idx is None:
                FLAGS.texture_conditional_channel_idx = [i for i in range(channels)]
            print(f"texture_conditional_channel_idx = {FLAGS.texture_conditional_channel_idx}")
            self.texture_conditional_channel_idx = FLAGS.texture_conditional_channel_idx
            self.texture_unconditional_channel_idx = [i for i in range(channels) if i not in self.texture_conditional_channel_idx]
            self.texture_unconditional_output_dims = len(self.texture_unconditional_channel_idx)
            self.texture_conditional_output_dims = len(self.texture_conditional_channel_idx)
            self.zero_conditional_inputs = torch.zeros(self.texture_conditional_input_dims, dtype=torch.float32, device='cuda')
            # assert self.texture_unconditional_channel_idx  + self.texture_conditional_channel_idx == list(range(channels)), \
                # f"Directly concate the two parts. If the index are not continuous, please modify the code"

        elif self.use_texture_conditional_inputs is False:
            self.texture_conditional_input_dims = 0
            self.texture_unconditional_output_dims = self.channels
            self.texture_conditional_output_dims = 0
        else:
            raise ValueError("use_texture_conditional_inputs must be True or False, get {}".format(self.use_texture_conditional_inputs))

        self.conditonal_inputs = None

        use_view_direction = FLAGS.get("use_view_direction", False)
        additional_num_input_dims = 3 if use_view_direction is True else 0
        self.texture_conditional_input_dims = self.texture_conditional_input_dims + additional_num_input_dims
        print(f"use_view_direction: {use_view_direction}, additional_num_input_dims: {additional_num_input_dims}")

        # Setup MLP
        mlp_cfg = {
            "n_input_dims" : self.encoder.n_output_dims,
            "n_cond_input_dims": self.texture_conditional_input_dims,
            "n_output_dims" : self.texture_unconditional_output_dims,
            "n_cond_output_dims": self.texture_conditional_output_dims,
            "n_hidden_layers" : hidden,
            "n_neurons" : self.internal_dims
        }
        self.net = _MLP(mlp_cfg, gradient_scaling)
        print("Encoder output: %d dims" % (self.encoder.n_output_dims))



    # Sample texture at a given location
    def sample(self, texc):
        texc_shape = texc.shape
        _texc = (texc.view(-1, 3) - self.AABB[0][None, ...]) / (self.AABB[1][None, ...] - self.AABB[0][None, ...])  # [B, rast_H, rast_W, rast_dim (3)]
        _texc = torch.clamp(_texc, min=0, max=1)
        
        p_enc = self.encoder(_texc.contiguous())
        if self.use_texture_conditional_inputs is False:
            out, _ = self.net(p_enc)
        else:
            if self.conditonal_inputs is None:
                LOGGER.warning("use_texture_conditional_inputs is True, but conditonal_inputs is None")
                conditonal_inputs = self.zero_conditional_inputs[None, None, None, :].repeat(texc_shape[0], texc_shape[1], texc_shape[2], 1).view(-1, self.texture_conditional_input_dims)
            else:
                conditonal_inputs = self.conditonal_inputs[:, None, None, :].repeat(1, texc_shape[1], texc_shape[2], 1).view(-1, self.texture_conditional_input_dims)
            uncond_out, cond_out = self.net(p_enc, conditonal_inputs)

            out = torch.zeros((conditonal_inputs.shape[0], self.channels), dtype=torch.float32, device='cuda')
            if uncond_out is not None:
                out[..., self.texture_unconditional_channel_idx] = uncond_out
            if cond_out is not None:
                out[..., self.texture_conditional_channel_idx] = cond_out
            # static_out = self.net.forward(p_enc, self.zero_conditional_inputs[None, None, None, :].repeat(texc_shape[0], texc_shape[1], texc_shape[2], 1).view(-1, self.texture_conditional_input_dims))
            # dynamic_out = self.net.forward(p_enc, self.conditonal_inputs[:, None, None, :].repeat(1, texc_shape[1], texc_shape[2], 1).view(-1, self.texture_conditional_input_dims))
            # out = torch.zeros_like(static_out)
            # out[..., self.texture_unconditional_channel_idx] = static_out[..., self.texture_unconditional_channel_idx]
            # out[..., self.texture_conditional_channel_idx] = dynamic_out[..., self.texture_conditional_channel_idx]
            # out = torch.cat([static_out[..., self.texture_unconditional_channel_idx], dynamic_out[..., self.texture_conditional_channel_idx]], dim=-1)

        # Sigmoid limit and scale to the allowed range
        out = torch.sigmoid(out) * (self.min_max[1][None, :] - self.min_max[0][None, :]) + self.min_max[0][None, :]

        return out.view(*texc.shape[:-1], self.channels) # Remap to [n, h, w, c]

    # In-place clamp with no derivative to make sure values are in valid range after training
    def clamp_(self):
        pass

    def cleanup(self):
        tcnn.free_temporary_memory()

    def register_conditonal_inputs(self, inputs):
        self.conditonal_inputs = inputs  # must be [N, num_dims]

