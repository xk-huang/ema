# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

from typing import Dict, Optional, Union
import torch
import xatlas
import numpy as np
import logging

from render import mesh
from render import render
from render import regularizer
from render.util import safe_normalize, dot, rgb_to_srgb
from .skin_net import get_skinning_weight_net
from utils.bones import sample_on_bone_head_and_tail
from .non_rigid_offset_net import get_non_rigid_offset_net

LOGGER = logging.getLogger(__name__)


# [FIXME](xk): still have bugs for rendering texture


###############################################################################
#  Geometry interface
###############################################################################

class DLMeshDyn(torch.nn.Module):
    def __init__(self, *, initial_guess: Union[mesh.Mesh, Dict], animation_meta_data: Dict, learn_skinning, learn_non_rigid_offset: bool, FLAGS):
        super(DLMeshDyn, self).__init__()

        self.FLAGS = FLAGS
        if FLAGS.local_rank == 0:
            print = LOGGER.info
        else:
            print = LOGGER.debug

        if not isinstance(initial_guess, mesh.Mesh):
            raise ValueError(f"initial_guess must be a mesh.Mesh or a dict, got {type(initial_guess)}")

        if initial_guess.v_tex is None or initial_guess.t_tex_idx is None:
            print("Initial guess has no texture uv, parametrize it with xatlas.")
            _, indices, uvs = xatlas.parametrize(initial_guess.v_pos, initial_guess.t_pos_idx)
            indices_int64 = indices.astype(np.uint64, casting='same_kind').view(np.int64)

            uvs = torch.tensor(uvs, dtype=torch.float32)
            faces = torch.tensor(indices_int64, dtype=torch.int64)
            initial_guess.v_tex = uvs
            initial_guess.t_tex_idx = faces

        initial_guess = initial_guess.to("cuda")

        self.initial_guess = initial_guess
        self.mesh          = initial_guess.clone()
        print("Base mesh has %d triangles and %d vertices." % (self.mesh.t_pos_idx.shape[0], self.mesh.v_pos.shape[0]))
        
        self.mesh.v_pos = torch.nn.Parameter(self.mesh.v_pos, requires_grad=True)
        self.register_parameter('vertex_pos', self.mesh.v_pos)

        # [NOTE] scale and center are very important for skinning learning,
        # which make sure the range of query points are correct.
        self.scale = torch.tensor(self.FLAGS.mesh_scale, dtype=torch.float32, device='cuda')
        self.center = torch.tensor(self.FLAGS.mesh_center, dtype=torch.float32, device='cuda')

        self.use_tet_aabb_in_mesh = FLAGS.use_tet_aabb_in_mesh
        if self.use_tet_aabb_in_mesh:
            # TODO: move AABB to self.AABB
            print("Use tetrahedron AABB for mesh")  
            if FLAGS.tet_dir is None:
                FLAGS.tet_dir = 'data/tets/'
            grid_res = FLAGS.dmtet_grid
            print(f"tet_dir = {FLAGS.tet_dir}, grid_res = {grid_res}")
            tets = np.load('{}/{}_tets.npz'.format(FLAGS.tet_dir, grid_res))
            verts = torch.tensor(tets['vertices'], dtype=torch.float32, device='cuda')
            scale = torch.tensor(FLAGS.mesh_scale, dtype=torch.float32, device='cuda')
            verts = verts * scale
            self.tet_aabb = (torch.min(verts, dim=0).values, torch.max(verts, dim=0).values)

        self.AABB = self.getAABB()
        print(f"Geometry AABB: {self.AABB}")

        self.learn_skinning = learn_skinning
        self.lbs_weights: Optional[torch.Tensor]
        self.skin_net: Optional[torch.nn.Module] = None
        # self.num_joints = rest_smpl_data["rest_joints_in_canon"].shape[0]
        self.num_uniq_tfs = animation_meta_data["num_uniq_tfs"]
        self.rest_tfs = animation_meta_data["rest_tfs_global"]

        if learn_skinning:
            print("Learn skinning weights")
            self.lbs_weights = None
            self.skin_net_encoding_type = FLAGS.skin_net_encoding_type

            if self.skin_net_encoding_type == 'meta_skin_net':
                if FLAGS.tfs_type == 'bone':
                    raise ValueError(f"Meta skinning network is not supported for bone transformation, tfs_type shoule be \"joint\".")
                print("Use meta skinning network")
                from .meta_skin_net import get_meta_skinning_weight_net
                from .meta_skin_net import hierarchical_softmax
                self.skin_net = get_meta_skinning_weight_net(FLAGS.meta_skin_net_weight_path)  # TODO: add meta skinning network
                self.hierarchical_softmax = hierarchical_softmax
            else:
                self.skin_net = get_skinning_weight_net(self.num_uniq_tfs, self.skin_net_encoding_type, FLAGS.skin_net_num_freq)

            self.skin_net_logit_softmax_temperature: int = FLAGS.skin_net_logit_softmax_temperature

        else:
            print("Use SMPL/learned skinning weights")
            self.register_buffer('lbs_weights', animation_meta_data["lbs_weights"], persistent=False) # (6890, 24)
            self.skin_net = None

        if FLAGS.tfs_type == 'joint':
            self.tfs_type = "tfs_in_canon"
            print(f"Use pose transformation based on joint: {self.tfs_type}")
        elif FLAGS.tfs_type == 'bone':
            self.tfs_type = "tfs_bone_in_canon"
            print(f"Use pose transformation based on bone (previous joint): {self.tfs_type}")
        else:
            raise ValueError(f"Unknown tfs_type: {FLAGS.tfs_type}")

        bone_heads = animation_meta_data["bone_heads"]
        bone_tails = animation_meta_data["bone_tails"]
        bone2uniq_tf = animation_meta_data["bone2uniq_tf"]
        self.register_buffer('bone_heads', bone_heads, persistent=False) # (24, 3)
        self.register_buffer('bone_tails', bone_tails, persistent=False) # (24, 3)
        self.register_buffer('bone2uniq_tf', bone2uniq_tf, persistent=False) # (24, 3)
        print(f"Number of unique transformations: {self.num_uniq_tfs}")

        self.learn_non_rigid_offset = learn_non_rigid_offset
        if learn_non_rigid_offset is True:
            non_rigid_offset_net_encoding_type = FLAGS.non_rigid_offset_net_encoding_type
            non_rigid_offset_input_dim = FLAGS.non_rigid_offset_input_dim

            print(f"Enable non-rigid offset: encoding_type {non_rigid_offset_net_encoding_type}, input_dim {non_rigid_offset_input_dim}")

            self.non_rigid_offset_net = get_non_rigid_offset_net(3, FLAGS.non_rigid_offset_input_dim, FLAGS.non_rigid_offset_net_encoding_type, FLAGS.non_rigid_offset_net_num_freq)
            
            self.non_rigid_type = FLAGS.non_rigid_type
            assert self.non_rigid_type in ['canon', 'world'], f"Unknown non_rigid_type: {self.non_rigid_type}"
        else:
            print("Disable non-rigid offset")
            self.non_rigid_offset_net = None
            self.non_rigid_type = None

        self.to("cuda")

        self.return_kd_grad = (self.FLAGS.albedo_regularizer > 0)
        print(f"Return kd grad: {self.return_kd_grad}")

    def query_weights(self, x_cano):
        if self.skin_net_encoding_type == 'meta_skin_net':
            correct_range_x_cano = x_cano # [-1, 1]
            correct_range_x_cano = torch.clamp(correct_range_x_cano, min=-1, max=1)
        else:
            # [NOTE] use AABB to remap query ranges
            # correct_range_x_cano = (x_cano - self.center) / self.scale  # [-0.5, 0.5], centered at [0.0]
            correct_range_x_cano = (x_cano - self.AABB[0]) / (self.AABB[1] - self.AABB[0])  # [0.0, 1.0]
            correct_range_x_cano = torch.clamp(correct_range_x_cano, min=0, max=1)

        weights_logit = self.skin_net(correct_range_x_cano).to(correct_range_x_cano.dtype)  # (N, J)
        if torch.is_anomaly_enabled():
            assert torch.all(torch.isfinite(weights_logit))

        if self.skin_net_encoding_type == 'meta_skin_net':
            weights = self.hierarchical_softmax(weights_logit / self.skin_net_logit_softmax_temperature)  # hierarchical softmax in SNARF, [N, 24]
        else:
            weights = torch.functional.F.softmax(weights_logit / self.skin_net_logit_softmax_temperature, dim=-1)
        return weights, weights_logit

    @torch.no_grad()
    def getAABB(self):
        if self.use_tet_aabb_in_mesh:
            return self.tet_aabb
        return mesh.aabb(self.mesh)

    def updatePosedMesh(self, mesh: mesh.Mesh, target=None, update_weights=True, add_non_rigid_offsets=True):
        # XXX(xiaoke): `add_non_rigid_offsets`, `FLAGS.learn_non_rigid_offset`, and `self.non_rigid_type` are confusing.
        # `add_non_rigid_offsets` is used to split the non-rigid offset in the training process.
        # `FLAGS.learn_non_rigid_offset` controls whether to init non-rigid network, and assigns the non-rigid type to "world", "canon", or None.
        if target is None:
            return mesh

        # forward skinning
        tfs_in_canon = target[self.tfs_type].to("cuda")  # (N, 24, 4, 4)
        if self.learn_skinning:
            if update_weights or self.lbs_weights is None or self.lbs_weights.shape[0] != mesh.v_pos.shape[0]:
                LOGGER.debug("update weights in `updatePosedMesh`")
                self.lbs_weights, self.lbs_weights_logits = self.query_weights(mesh.v_pos)
                self.lbs_weights, self.lbs_weights_logits = self.lbs_weights.float(), self.lbs_weights_logits.float()
            
            T = torch.einsum("ij,bjkl->bikl", self.lbs_weights, tfs_in_canon)  # (N, 6890, 4, 4)
            
        else:
            T = torch.einsum("ij,bjkl->bikl", self.lbs_weights, tfs_in_canon)  # (N, 6890, 4, 4)

        # non rigid offset 
        if self.non_rigid_offset_net is not None:
            non_rigid_offsets = self.non_rigid_offset_net(mesh.v_pos[None], target["params"])  # (B, V, 3)
            if self.FLAGS.non_rigid_offset_smooth is True:
                _non_rigid_offset_mesh = Meshes(verts=non_rigid_offsets, faces=self.dmtet_faces.unsqueeze(0).expand(non_rigid_offsets.shape[0], -1, -1))
                _non_rigid_offset_mesh = taubin_smoothing(_non_rigid_offset_mesh, num_iter=10)
                non_rigid_offsets = _non_rigid_offset_mesh.verts_padded()
            self.non_rigid_offsets = non_rigid_offsets

        batch_size = len(tfs_in_canon)

        # [NOTE] query color in canonical space. No non-rigid offsets!
        mesh.v_pos_canon = mesh.v_pos = mesh.v_pos.unsqueeze(0) # (1, 6890, 3)

        if add_non_rigid_offsets is True and self.non_rigid_type == 'canon':
            mesh.v_pos = mesh.v_pos + non_rigid_offsets

        homo_v_pos = torch.functional.F.pad(mesh.v_pos, (0,1), value=1.0)
        batched_homo_v_pos = homo_v_pos.expand(batch_size, -1, -1)  # (N, 6890, 4)
        posed_batched_homo_v_pos = torch.einsum("bij,bikj->bik", batched_homo_v_pos, T) # (N, 6890, 4)
        mesh.v_pos = posed_batched_homo_v_pos[..., :3]
        
        if add_non_rigid_offsets is True and self.non_rigid_type == 'world':
            mesh.v_pos = mesh.v_pos + non_rigid_offsets

        # mesh.v_tex = mesh.v_tex.unsqueeze(0).expand(batch_size, -1, -1)
        mesh.v_tex = mesh.v_tex.unsqueeze(0) # (1, 6890, 3)

        return mesh

    def getMesh(self, material, target=None):
        self.mesh.material = material

        imesh = mesh.Mesh(base=self.mesh)
        # Compute normals and tangent space
        if target is None:
            imesh = mesh.auto_normals(imesh)
            imesh = mesh.compute_tangents(imesh)
            imesh.v_pos_canon = imesh.v_pos
            return imesh

        # [FIXME] unchecked code, maybe the transformed normal and tangent are not correct.
        imesh = self.updatePosedMesh(imesh, target)
        imesh = mesh.batched_auto_normals(imesh)
        imesh = mesh.batched_compute_tangents(imesh)

        return imesh

    def render(self, glctx, target, lgt, opt_material, bsdf=None, return_kd_grad=True, **kwargs):
        bsdf = self.FLAGS.bsdf if bsdf is None else bsdf
        opt_mesh = self.getMesh(opt_material, target)
        if 'kd_ks_normal' in opt_material and opt_material['kd_ks_normal'].use_texture_conditional_inputs is True:
            opt_material['kd_ks_normal'].register_conditonal_inputs(target['params'])
        return render.render_mesh(self.FLAGS, glctx, opt_mesh, target['mvp'], target['campos'], lgt, target['resolution'], spp=self.FLAGS.spp, 
                                    num_layers=self.FLAGS.layers, msaa=True, background=target['background'], bsdf=bsdf, return_kd_grad=return_kd_grad)

    def tick(self, glctx, target, lgt, opt_material, loss_fn, iteration, lpips_loss_fn):

        # ==============================================================================================
        #  Render optimizable object with identical conditions
        # ==============================================================================================
        buffers = self.render(glctx, target, lgt, opt_material, return_kd_grad=self.return_kd_grad)

        # ==============================================================================================
        #  Compute loss
        # ==============================================================================================
        t_iter = iteration / self.FLAGS.iter  # [0, 1], for reg weighting

        # Image-space loss, split into a coverage component and a color component
        color_ref = target['img']
        img_loss = torch.nn.functional.mse_loss(buffers['shaded'][..., 3:], color_ref[..., 3:]) 

        shaded_rgb = buffers['shaded'][..., 0:3] * color_ref[..., 3:]
        ref_rgb = color_ref[..., 0:3] * color_ref[..., 3:]
        img_loss = img_loss + loss_fn(shaded_rgb, ref_rgb)

        if lpips_loss_fn is not None:
            if self.FLAGS.lpips_in_srgb:
                color_ref, ref_rgb = rgb_to_srgb(color_ref), rgb_to_srgb(ref_rgb)
            if self.FLAGS.lpips_regularizer > 0:
                img_loss = img_loss + self.FLAGS.lpips_regularizer * lpips_loss_fn.forward(shaded_rgb.permute(0, 3, 1, 2), ref_rgb.permute(0, 3, 1, 2)).mean()

        reg_losses = {}

        # Compute regularizer. 
        if self.FLAGS.use_training_tricks:
            laplace_weight = self.FLAGS.laplace_scale * (1 - t_iter)
            albedo_weight = self.FLAGS.albedo_regularizer * min(1.0, iteration / 500)
            visibility_weight =  self.FLAGS.visibility_regularizer * min(1.0, iteration / 500)
        else:
            laplace_weight = self.FLAGS.laplace_scale
            albedo_weight = self.FLAGS.albedo_regularizer
            visibility_weight =  self.FLAGS.visibility_regularizer
        light_weight = self.FLAGS.light_regularizer
        skinning_weight = self.FLAGS.skinning_regularizer
        offset_weight = self.FLAGS.non_rigid_offset_regularizer

        # Laplacian regularizer
        if self.FLAGS.laplace == "absolute":
            reg_losses["laplace"] = regularizer.laplace_regularizer_const(self.mesh.v_pos, self.mesh.t_pos_idx) * laplace_weight
        elif self.FLAGS.laplace == "relative":
            reg_losses["laplace"] = regularizer.laplace_regularizer_const(self.mesh.v_pos - self.initial_guess.v_pos, self.mesh.t_pos_idx) * laplace_weight

        # Albedo (k_d) smoothnesss regularizer
        if self.return_kd_grad:
            reg_losses["albedo_kd_grad"] = torch.mean(buffers['kd_grad'][..., :-1] * buffers['kd_grad'][..., -1:]) * albedo_weight

        # Visibility regularizer
        reg_losses["visibility_ao"] = torch.mean(buffers['occlusion'][..., :-1] * buffers['occlusion'][..., -1:]) * visibility_weight

        # Light white balance regularizer
        reg_losses["light"] = lgt.regularizer() * light_weight

        # bls weight and offset regularizer
        motion_losses = {"void": torch.zeros(1, device="cuda")}
        if self.learn_skinning:
            motion_losses = self.get_motion_losses()
            motion_losses["bone_skinning"] *= skinning_weight
        if self.learn_non_rigid_offset:
            motion_losses["bone_offset"] *=  offset_weight

        # lpips loss

        return img_loss, reg_losses, motion_losses, {"laplace_weight": laplace_weight, "albedo_weight": albedo_weight, "visibility_weight": visibility_weight, "light_weight": light_weight, "skinning_weight": skinning_weight, "offset_weight": offset_weight}

    def get_motion_losses(self):
        losses = {}

        bone_samples = sample_on_bone_head_and_tail(self.bone_heads, self.bone_tails, n_per_bone=5, range=(0.1, 0.9))
        bone_samples_shapes = bone_samples.shape
        bone_samples = bone_samples.reshape(-1, bone_samples_shapes[-1])

        weights, _ = self.query_weights(bone_samples)
        weights = weights.view(bone_samples_shapes[0], bone_samples_shapes[1], -1)
        weights_gt = torch.nn.functional.one_hot(self.bone2uniq_tf, num_classes=self.num_uniq_tfs).expand_as(weights).to(weights)
        losses['bone_skinning'] = torch.nn.functional.mse_loss(weights, weights_gt)

        if self.learn_non_rigid_offset:
            raise NotImplementedError

        return losses
