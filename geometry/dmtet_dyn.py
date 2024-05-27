# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import numpy as np
import torch
from typing import Optional, Dict
import logging
import os

from render import mesh
from render import render
from render import regularizer
from render.util import safe_normalize, dot, rgb_to_srgb
from .skin_net import get_skinning_weight_net
from utils.bones import sample_on_bone_head_and_tail
from .non_rigid_offset_net import get_non_rigid_offset_net
from .geometry_net import get_geometry_net
from utils.mesh_bbox_sdf import get_mesh_bbox_sdf
from pytorch3d.structures import Meshes
from pytorch3d.ops.mesh_filtering import taubin_smoothing
from utils.ops import sample_points_from_meshes


LOGGER = logging.getLogger(__name__)

def divide_no_nan(a, b):
    """
    Auxiliary funtion to handle divide by 0
    """
    div = a / b
    div[div != div] = 0.0
    div[div == float('inf')] = 0.0
    return div

###############################################################################
# Marching tetrahedrons implementation (differentiable), adapted from
# https://github.com/NVIDIAGameWorks/kaolin/blob/master/kaolin/ops/conversions/tetmesh.py
###############################################################################

class DMTetDyn:
    def __init__(self):
        self.triangle_table = torch.tensor([
                [-1, -1, -1, -1, -1, -1],
                [ 1,  0,  2, -1, -1, -1],
                [ 4,  0,  3, -1, -1, -1],
                [ 1,  4,  2,  1,  3,  4],
                [ 3,  1,  5, -1, -1, -1],
                [ 2,  3,  0,  2,  5,  3],
                [ 1,  4,  0,  1,  5,  4],
                [ 4,  2,  5, -1, -1, -1],
                [ 4,  5,  2, -1, -1, -1],
                [ 4,  1,  0,  4,  5,  1],
                [ 3,  2,  0,  3,  5,  2],
                [ 1,  3,  5, -1, -1, -1],
                [ 4,  1,  2,  4,  3,  1],
                [ 3,  0,  4, -1, -1, -1],
                [ 2,  0,  1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1]
                ], dtype=torch.long, device='cuda')

        self.num_triangles_table = torch.tensor([0,1,1,2,1,2,2,1,1,2,2,1,2,1,1,0], dtype=torch.long, device='cuda')
        self.base_tet_edges = torch.tensor([0,1,0,2,0,3,1,2,1,3,2,3], dtype=torch.long, device='cuda')

    ###############################################################################
    # Utility functions
    ###############################################################################

    def sort_edges(self, edges_ex2):
        with torch.no_grad():
            order = (edges_ex2[:,0] > edges_ex2[:,1]).long()
            order = order.unsqueeze(dim=1)

            a = torch.gather(input=edges_ex2, index=order, dim=1)      
            b = torch.gather(input=edges_ex2, index=1-order, dim=1)  

        return torch.stack([a, b],-1)

    def map_uv(self, faces, face_gidx, max_idx):
        N = int(np.ceil(np.sqrt((max_idx+1)//2)))
        tex_y, tex_x = torch.meshgrid(
            torch.linspace(0, 1 - (1 / N), N, dtype=torch.float32, device="cuda"),
            torch.linspace(0, 1 - (1 / N), N, dtype=torch.float32, device="cuda"),
            indexing='ij'
        )

        pad = 0.9 / N

        uvs = torch.stack([
            tex_x      , tex_y,
            tex_x + pad, tex_y,
            tex_x + pad, tex_y + pad,
            tex_x      , tex_y + pad
        ], dim=-1).view(-1, 2)

        def _idx(tet_idx, N):
            x = tet_idx % N
            y = torch.div(tet_idx, N, rounding_mode='trunc')
            return y * N + x

        tet_idx = _idx(torch.div(face_gidx, 2, rounding_mode='trunc'), N)
        tri_idx = face_gidx % 2

        uv_idx = torch.stack((
            tet_idx * 4, tet_idx * 4 + tri_idx + 1, tet_idx * 4 + tri_idx + 2
        ), dim = -1). view(-1, 3)

        return uvs, uv_idx

    ###############################################################################
    # Marching tets implementation
    ###############################################################################

    def __call__(self, pos_nx3, sdf_n, tet_fx4):
        with torch.no_grad():
            occ_n = sdf_n > 0 # (V), [0, 1]
            occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1,4) # (T, 4), range [0, 1]
            occ_sum = torch.sum(occ_fx4, -1)  # (T), range [0, 1, 2]
            valid_tets = (occ_sum>0) & (occ_sum<4)  # (T_valid), range [True, False]
            occ_sum = occ_sum[valid_tets]  # (T_valid), range [0, 1, 2]

            # find all vertices
            all_edges = tet_fx4[valid_tets][:,self.base_tet_edges].reshape(-1,2)  # (T_valid*6, 2), range [0, V-1]
            all_edges = self.sort_edges(all_edges)  # (T_valid*6, 2), range [0, V-1]
            unique_edges, idx_map = torch.unique(all_edges,dim=0, return_inverse=True)  # (E_unique, 2), range [0, V-1]; (T_valid*6), range [0, E_unique-1]
            
            unique_edges = unique_edges.long()  # (E_unique, 2), range [0, V-1]
            mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1,2).sum(-1) == 1  # (E_unique), range [True, False], whether it is cut edge or not
            mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device="cuda") * -1  # (E_unique), range [-1]
            mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.long,device="cuda")  # (E_unique), range [-1, E_cut-1], unique edge index to cut edge index
            idx_map = mapping[idx_map] # map edges to verts. (T_valid*6), range [-1, E_cut-1]

            interp_v = unique_edges[mask_edges] # (E_cut, 2), range [0, V-1]
        edges_to_interp = pos_nx3[interp_v.reshape(-1)].reshape(-1,2,3)
        edges_to_interp_sdf = sdf_n[interp_v.reshape(-1)].reshape(-1,2,1)
        edges_to_interp_sdf[:,-1] *= -1

        denominator = edges_to_interp_sdf.sum(1,keepdim = True)

        edges_to_interp_sdf = torch.flip(edges_to_interp_sdf, [1])/denominator
        verts = (edges_to_interp * edges_to_interp_sdf).sum(1)

        idx_map = idx_map.reshape(-1,6)  # (T_valid, 6), range [-1, E_cut-1]

        # Marching tets
        v_id = torch.pow(2, torch.arange(4, dtype=torch.long, device="cuda"))
        tetindex = (occ_fx4[valid_tets] * v_id.unsqueeze(0)).sum(-1)  # (T_valid), range [0, 15]
        num_triangles = self.num_triangles_table[tetindex]

        # Generate triangle indices
        faces = torch.cat((
            torch.gather(input=idx_map[num_triangles == 1], dim=1, index=self.triangle_table[tetindex[num_triangles == 1]][:, :3]).reshape(-1,3),
            torch.gather(input=idx_map[num_triangles == 2], dim=1, index=self.triangle_table[tetindex[num_triangles == 2]][:, :6]).reshape(-1,3),
        ), dim=0)

        # Get global face index (static, does not depend on topology)
        num_tets = tet_fx4.shape[0]
        tet_gidx = torch.arange(num_tets, dtype=torch.long, device="cuda")[valid_tets]  # (T_valid), range [0, T-1]
        face_gidx = torch.cat((
            tet_gidx[num_triangles == 1]*2,
            torch.stack((tet_gidx[num_triangles == 2]*2, tet_gidx[num_triangles == 2]*2 + 1), dim=-1).view(-1)
        ), dim=0)

        # [FIXME] unique the uvs and uv_idx, the uvs are too many there, 20M.
        uvs, uv_idx = self.map_uv(faces, face_gidx, num_tets*2)

        # return tet_idx for each triangle
        tet_idx = torch.arange(tet_fx4.shape[0], device="cuda")[valid_tets]
        tet_idx = torch.cat((tet_idx[num_triangles == 1], tet_idx[num_triangles ==2].unsqueeze(-1).expand(-1, 2).reshape(-1)), dim=0)

        return verts, faces, uvs, uv_idx, valid_tets, tet_idx

###############################################################################
# Regularizer
###############################################################################

def sdf_reg_loss(sdf, all_edges):
    sdf_f1x6x2 = sdf[all_edges.reshape(-1)].reshape(-1,2)
    mask = torch.sign(sdf_f1x6x2[...,0]) != torch.sign(sdf_f1x6x2[...,1])
    sdf_f1x6x2 = sdf_f1x6x2[mask]
    sdf_diff = torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[...,0], (sdf_f1x6x2[...,1] > 0).float()) + \
            torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[...,1], (sdf_f1x6x2[...,0] > 0).float())
    return sdf_diff

###############################################################################
#  Geometry interface
###############################################################################

class DMTetGeometryDyn(torch.nn.Module):
    def __init__(self, *, grid_res, animation_meta_data: Dict, learn_skinning: bool, learn_non_rigid_offset: bool, FLAGS):
        super(DMTetGeometryDyn, self).__init__()

        self.FLAGS         = FLAGS
        if FLAGS.local_rank == 0:
            print = LOGGER.info
        else:
            print = LOGGER.debug

        # [XXX] the grid_res should be halfed if permanent subdivision (no grad) is used.
        grid_res = torch.tensor(grid_res, device='cuda')
        self.register_buffer("grid_res", grid_res)
        self.marching_tets = DMTetDyn()
        self.base_tet_edges = self.marching_tets.base_tet_edges
        self.sort_edges = self.marching_tets.sort_edges

        if FLAGS.tet_dir is None:
            FLAGS.tet_dir = 'data/tets/'
        print(f"tet_dir = {FLAGS.tet_dir}, grid_res = {grid_res}")
        tets = np.load('{}/{}_tets.npz'.format(FLAGS.tet_dir, self.grid_res))

        # [FIXME] clarify the name: tet_verts, tet_indices. refer to `self.update_base_mesh`
        # The range of tet grid is [-0.5, 0.5], centered at [0.0]
        verts    = torch.tensor(tets['vertices'], dtype=torch.float32, device='cuda')
        self.register_buffer('verts', verts)
        indices  = torch.tensor(tets['indices'], dtype=torch.long, device='cuda')
        self.register_buffer('indices', indices)
        self.generate_edges()

        # [NOTE] scale and center are very important for skinning learning,
        # which make sure the range of query points are correct.
        self.scale = torch.tensor(self.FLAGS.mesh_scale, dtype=torch.float32, device='cuda')
        self.verts = self.verts * self.scale
        print(f"Scaling mesh to {self.verts.min(dim=0)[0]} - {self.verts.max(dim=0)[0]}")

        self.center = torch.tensor(self.FLAGS.mesh_center, dtype=torch.float32, device='cuda')
        self.verts = self.verts + self.center
        print(f"Centering mesh to {self.verts.mean(dim=0)}")

        self.AABB = self.getAABB()
        print(f"Geometry AABB: {self.AABB}")

        # Random init
        if FLAGS.learn_sdf_with_mlp is True:
            self.sdf = None

            self.learn_tet_vert_deform_with_mlp = FLAGS.learn_tet_vert_deform_with_mlp
            if not self.learn_tet_vert_deform_with_mlp:
                self.deform = torch.nn.Parameter(torch.zeros_like(self.verts), requires_grad=True)
                self.register_parameter('deform', self.deform)
            else:
                self.deform = None

            self.geometry_net = get_geometry_net(FLAGS.learn_tet_vert_deform_with_mlp, FLAGS.sdf_mlp_type, FLAGS.sdf_mlp_num_freq)

        elif FLAGS.learn_sdf_with_mlp is False:
            sdf = torch.rand_like(self.verts[:,0]) - 0.1  # [-0.1, 0.9], 10% inside the mesh, 90% outside the mesh.

            self.sdf    = torch.nn.Parameter(sdf.clone().detach(), requires_grad=True)
            self.register_parameter('sdf', self.sdf)

            self.deform = torch.nn.Parameter(torch.zeros_like(self.verts), requires_grad=True)
            self.register_parameter('deform', self.deform)

            self.geometry_net = None
        else:
            raise ValueError(f"Invalid learn_sdf_with_mlp, type {type(FLAGS.learn_sdf_with_mlp)}")
        self.dmtet_verts = None
        self.dmtet_faces = None
        self.dmtet_uvs = None
        self.dmtet_uv_idx = None
        self.valid_tets = None
        self.tet_verts_deformed = None

        self.learn_skinning = learn_skinning
        self.lbs_weights: Optional[torch.Tensor]
        self.skin_net: Optional[torch.nn.Module] = None
        # self.num_joints = rest_smpl_data["rest_joints_in_canon"].shape[0]
        self.num_uniq_tfs = animation_meta_data["num_uniq_tfs"]

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
            raise ValueError("SMPL skinning weight is used for fixed topology with 6890 points, DMTet change topology.")
            print("Use SMPL skinning weights")
            self.register_buffer('lbs_weights', animation_meta_data["lbs_weights"], persistent=False) # (6890, 24)
        self.lbs_weights = None

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

        self.regularize_sdf = (FLAGS.bone_sdf_regularizer > 0)
        if self.regularize_sdf:
            self.bone_sdf_regularizer = FLAGS.bone_sdf_regularizer
            self.bone_sdf_hinge = FLAGS.bone_sdf_hinge
            print(f"regularize_sdf: {self.regularize_sdf}, weight: {FLAGS.bone_sdf_regularizer}, hinge {FLAGS.bone_sdf_hinge}")
        else:
            print(f"regularize_sdf: {self.regularize_sdf}")

        self.return_kd_grad = any([_reg > 0 for _reg in (FLAGS.albedo_regularizer, FLAGS.ks_regularizer)])
        print(f"Return kd grad: {self.return_kd_grad}")

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

        if FLAGS.use_loss_scaling is True:
            self.num_loss_scaling_buffers = FLAGS.num_loss_scaling_buffers
            self.loss_scaling_type = FLAGS.loss_scaling_type
            self.register_buffer('loss_scaling_buffers', torch.ones(self.num_loss_scaling_buffers) * 10)
            self.register_buffer('loss_scaling_idx', torch.tensor(0))
            print(f"Use loss scaling: length: {self.num_loss_scaling_buffers}")

            assert self.loss_scaling_type in ["full", "image", "mask"], f"Unknown loss_scaling_type: {self.loss_scaling_type}"
        else:
            self.loss_scaling_buffers = None

        # create smpl mesh
        if FLAGS.smpl_surface_skinning_regularizser > 0:
            if FLAGS.tfs_type == 'bone':
                raise ValueError(f"SMPL surface skinning regularizer is not supported for bone transformation, tfs_type shoule be \"joint\".")

            print("Use SMPL mesh for surface skinning regularizer")
            _rest_verts_in_canon = animation_meta_data["rest_verts_in_canon"].unsqueeze(0)
            _faces = animation_meta_data["faces"].unsqueeze(0)

            self.smpl_mesh_pt3d = Meshes(_rest_verts_in_canon, _faces).cuda()
            self.smpl_lbs_weights = animation_meta_data["lbs_weights"].cuda()
        else:
            self.smpl_mesh_pt3d = None

        self.to("cuda")

        if self.geometry_net is not None and "windowed" in FLAGS.sdf_mlp_type:
            print("Geometry Net: set up windowed positional encoding")
            print(f"num_base_freq: {self.FLAGS.sdf_mlp_num_base_freq}")
            if self.FLAGS.sdf_mlp_num_base_freq >= self.geometry_net[0].num_frequencies:
                raise ValueError(f"sdf_mlp_num_base_freq {self.FLAGS.sdf_mlp_num_base_freq} >= num_frequencies {self.geometry_net[0].num_frequencies}, fall back to full encoding")
            self.geometry_net[0].register_num_base_frequencies(self.FLAGS.sdf_mlp_num_base_freq)

        if self.skin_net is not None and "windowed" in FLAGS.skin_net_encoding_type:
            print("Skin Net: set up windowed positional encoding")
            print(f"num_base_freq: {self.FLAGS.skin_net_num_base_freq}")
            if self.FLAGS.skin_net_num_base_freq >= self.skin_net[0].num_frequencies:
                raise ValueError(f"skin_net_num_base_freq {self.FLAGS.skin_net_num_base_freq} >= num_frequencies {self.skin_net[0].num_frequencies}, fall back to full encoding")

            self.skin_net[0].register_num_base_frequencies(self.FLAGS.skin_net_num_base_freq)

        if self.non_rigid_offset_net is not None and "windowed" in FLAGS.non_rigid_offset_net_encoding_type:
            print("Non Rigid Net: set up windowed positional encoding")
            print(f"num_base_freq: {self.FLAGS.non_rigid_offset_net_num_base_freq}")
            if self.FLAGS.non_rigid_offset_net_num_base_freq >= self.non_rigid_offset_net[0].num_frequencies:
                raise ValueError(f"non_rigid_offset_net_num_base_freq {self.FLAGS.non_rigid_offset_net_num_base_freq} >= num_frequencies {self.non_rigid_offset_net[0].num_frequencies}, fall back to full encoding")

            self.non_rigid_offset_net[0].register_num_base_frequencies(self.FLAGS.non_rigid_offset_net_num_base_freq)

        if FLAGS.no_train is True:
            print("No train, not pre-train each module.")
            return

        if FLAGS.pre_train_tet_with_bone_capsule is True:
            print("Pre-train tet with bone capsule")
            self.pre_train_tet_with_bone_capsule()
        else:
            self.init_sdf = None

        if FLAGS.pre_train_skin_net_with_bone_capsule is True:
            print("Pre-train skin net with bone capsule")
            self.pre_train_skin_net_with_bone_capsule()
        
        if self.learn_non_rigid_offset is True and FLAGS.pre_train_non_rigid_offset_net_with_bone_capsule is True:
            print("Pre-train non-rigid offset net with bone capsule")
            self.pre_train_non_rigid_offset_net_with_bone_capsule()

        if FLAGS.pre_train_with_smpl is True:
            print("Pre-train with SMPL")
            self.pre_train_with_smpl(animation_meta_data)
        else:
            self.init_sdf = None

    def pre_train_with_smpl(self, animation_meta_data):
        import trimesh

        if self.FLAGS.local_rank == 0:
            print = LOGGER.info
        else:
            print = LOGGER.debug

        rest_verts_in_canon = animation_meta_data["rest_verts_in_canon"].cuda()
        faces = animation_meta_data["faces"]

        # Pre-train SDF
        print("Pre-train with SMPL: get init sdf")
        smpl_mesh = trimesh.Trimesh(vertices=rest_verts_in_canon.cpu().numpy(), faces=faces.cpu().numpy())

        init_sdf_path = os.path.join(self.FLAGS.tet_dir, f"{self.grid_res}_init_sdf.smpl.{'vitruvian_pose' if self.FLAGS.use_vitruvian_pose else 't_pose'}.npz")

        if self.FLAGS.multi_gpu:
            if self.FLAGS.local_rank == 0 and (not os.path.exists(init_sdf_path)):
                init_sdf = self.create_init_sdf(smpl_mesh, print)
                self.save_init_sdf(init_sdf_path, init_sdf, print)
            else:
                torch.distributed.barrier(device_ids=[self.FLAGS.local_rank])

        if os.path.exists(init_sdf_path):
            print(f"Load pre-trained sdf from {init_sdf_path}")
            init_sdf = np.load(init_sdf_path)["sdf"]
        else:
            init_sdf = self.create_init_sdf(smpl_mesh, print)
            self.save_init_sdf(init_sdf_path, init_sdf, print)

        init_sdf = torch.tensor(init_sdf, dtype=torch.float32, device='cuda')

        if self.geometry_net is None:
            self.sdf.data = init_sdf.data
        else:
            _geometry_net_optim = torch.optim.Adam(self.geometry_net.parameters(), lr=self.FLAGS.pre_train_sdf_mlp_learning_rate)

            sdf_mlp_lipschitz_weight = self.FLAGS.sdf_mlp_lipschitz_regularizer
            num_sdf_pre_train_samples = 1000000
            for i in range(self.FLAGS.pre_train_sdf_mlp_steps):
                _geometry_net_optim.zero_grad()
                self.update_sdf_and_deform()

                if self.sdf.shape[0] > num_sdf_pre_train_samples:
                    sdf_indices = torch.randperm(self.sdf.shape[0])[:num_sdf_pre_train_samples]
                    _sdf = self.sdf[sdf_indices]
                    _init_sdf = init_sdf[sdf_indices]
                else:
                    _sdf = self.sdf
                    _init_sdf = init_sdf

                loss = torch.mean((_sdf - _init_sdf) ** 2)
                if self.FLAGS.eikonal_regularizer > 0:
                    loss = loss + self.FLAGS.eikonal_regularizer * self.get_eikonal_loss(self._geometry_net_inputs, self.sdf)

                if self.geometry_net is not None  and'lipschitz' in self.FLAGS.sdf_mlp_type and sdf_mlp_lipschitz_weight > 0:
                    loss = loss + sdf_mlp_lipschitz_weight * self.geometry_net[1].get_lipschitz_loss()

                loss.backward()
                _geometry_net_optim.step()

                if i % 100 == 0:
                    print(f"Pre-train sdf mlp, step {i}, loss {loss.item()}")

            del _geometry_net_optim
            torch.cuda.empty_cache()


        # Pre-train Skinning
        if self.skin_net_encoding_type == "meta_skin_net":
            print("Use meta skin net trained on CAPE dataset, skip pre-train skin net with SMPL")
            return

        print("Pre-train with SMPL: get init lbs weights")
        _lbs_weights = animation_meta_data["lbs_weights"].cuda()
        if self.FLAGS.tfs_type == "bone":
            FROM_TIP_TO_INNER_JOINT = {
                11: 8,
                10: 7,
                23: 21,
                20: 22,
                15: 12,
            }
            for tip_id, inner_joint_id in FROM_TIP_TO_INNER_JOINT.items():
                _lbs_weights[:, inner_joint_id] = _lbs_weights[:, tip_id] + _lbs_weights[:, inner_joint_id]
            SELECTED_TF_IDS = [i for i in range(len(range(_lbs_weights.shape[1]))) if i not in FROM_TIP_TO_INNER_JOINT.keys()]
            _lbs_weights = _lbs_weights[:, SELECTED_TF_IDS]

        elif self.FLAGS.tfs_type == "joint":
            pass
        else:
            raise ValueError(f"Unknown tfs type {self.FLAGS.tfs_type}")

        _skin_net_optim = torch.optim.Adam(self.skin_net.parameters(), lr=self.FLAGS.pre_train_skin_net_learning_rate)

        skin_net_lipschitz_weight = self.FLAGS.skin_net_lipschitz_regularizer
        for i in range(self.FLAGS.pre_train_skin_net_steps):
            _skin_net_optim.zero_grad()
            _weights, _ = self.query_weights(rest_verts_in_canon)
            _skin_net_loss = torch.nn.functional.mse_loss(_weights, _lbs_weights)

            if 'lipschitz' in self.FLAGS.skin_net_encoding_type and skin_net_lipschitz_weight > 0:
                _skin_net_loss = _skin_net_loss + skin_net_lipschitz_weight * self.skin_net[1].get_lipschitz_loss()
        
            _skin_net_loss.backward()
            _skin_net_optim.step()
            if i % 100 == 0:
                print(f"Pre-train skin net with bone capsule, iter {i}, loss {_skin_net_loss.item()}")
        del _skin_net_optim
        torch.cuda.empty_cache()

        self.init_sdf = init_sdf.clone()

    def pre_train_non_rigid_offset_net_with_bone_capsule(self):
        _non_rigid_offset_net_optim = torch.optim.Adam(self.non_rigid_offset_net.parameters(), lr=self.FLAGS.pre_train_non_rigid_offset_net_learning_rate)
        params = torch.zeros((1, self.FLAGS.non_rigid_offset_input_dim), dtype=torch.float32, device="cuda")
        for i in range(self.FLAGS.pre_train_non_rigid_offset_net_steps):
            _non_rigid_offset_net_optim.zero_grad()
            _non_rigid_offset_net_loss = self.get_bone_offset_loss(params)
            _non_rigid_offset_net_loss.backward()
            _non_rigid_offset_net_optim.step()
            if i % 100 == 0:
                print(f"Pre-train non-rigid offset net with bone capsule, iter {i}, loss {_non_rigid_offset_net_loss.item()}")

    def pre_train_skin_net_with_bone_capsule(self):
        if self.skin_net_encoding_type == "meta_skin_net":
            print("Use meta skin net trained on CAPE dataset, skip pre-train skin net with SMPL")
            return

        _skin_net_optim = torch.optim.Adam(self.skin_net.parameters(), lr=self.FLAGS.pre_train_skin_net_learning_rate)
        for i in range(self.FLAGS.pre_train_skin_net_steps):
            _skin_net_optim.zero_grad()
            _skin_net_loss = self.get_bone_skinning_loss()
            _skin_net_loss.backward()
            _skin_net_optim.step()
            if i % 100 == 0:
                print(f"Pre-train skin net with bone capsule, iter {i}, loss {_skin_net_loss.item()}")
        del _skin_net_optim
        torch.cuda.empty_cache()

    def pre_train_tet_with_bone_capsule(self):
        from utils.creation import create_bone_capsule
        init_sdf_path = os.path.join(self.FLAGS.tet_dir, f"{self.grid_res}_init_sdf.{'vitruvian_pose' if self.FLAGS.use_vitruvian_pose else 't_pose'}.npz")

        if self.FLAGS.local_rank == 0:
            print = LOGGER.info
        else:
            print = LOGGER.debug
        
        if self.FLAGS.multi_gpu:
            if self.FLAGS.local_rank == 0 and (not os.path.exists(init_sdf_path)):
                bone_capsule_mesh = create_bone_capsule(self.bone_heads, self.bone_tails)
                init_sdf = self.create_init_sdf(bone_capsule_mesh, print)
                self.save_init_sdf(init_sdf_path, init_sdf, print)
            else:
                torch.distributed.barrier(device_ids=[self.FLAGS.local_rank])

        if os.path.exists(init_sdf_path):
            print(f"Load pre-trained sdf from {init_sdf_path}")
            init_sdf = np.load(init_sdf_path)["sdf"]
        else:
            bone_capsule_mesh = create_bone_capsule(self.bone_heads, self.bone_tails)
            init_sdf = self.create_init_sdf(bone_capsule_mesh, print)
            self.save_init_sdf(init_sdf_path, init_sdf, print)

        init_sdf = torch.tensor(init_sdf, dtype=torch.float32, device='cuda')

        if self.geometry_net is None:
            self.sdf.data = init_sdf.data
        else:
            num_sdf_pre_train_samples = 1000000
            _geometry_net_optim = torch.optim.Adam(self.geometry_net.parameters(), lr=self.FLAGS.pre_train_sdf_mlp_learning_rate)
            for i in range(self.FLAGS.pre_train_sdf_mlp_steps):
                _geometry_net_optim.zero_grad()
                self.update_sdf_and_deform()
                if self.sdf.shape[0] > num_sdf_pre_train_samples:
                    sdf_indices = torch.randperm(self.sdf.shape[0])[:num_sdf_pre_train_samples]
                    _sdf = self.sdf[sdf_indices]
                    _init_sdf = init_sdf[sdf_indices]
                else:
                    _sdf = self.sdf
                    _init_sdf = init_sdf
                loss = torch.mean((_sdf - _init_sdf) ** 2)
                if self.FLAGS.eikonal_regularizer > 0:
                    loss = loss + self.FLAGS.eikonal_regularizer * self.get_eikonal_loss(self._geometry_net_inputs, self.sdf)
                loss.backward()
                _geometry_net_optim.step()

                if i % 100 == 0:
                    print(f"Pre-train sdf mlp, step {i}, loss {loss.item()}")

            del _geometry_net_optim
            torch.cuda.empty_cache()

        self.init_sdf = init_sdf.clone()

    def update_sdf_and_deform(self):
        if self.geometry_net is not None:
            query_verts = self.verts

            self._geometry_net_inputs = query_verts
            if self.FLAGS.eikonal_regularizer > 0:
                self._geometry_net_inputs.requires_grad = True

            query_verts = (query_verts - self.AABB[0]) / (self.AABB[1] - self.AABB[0])
            query_verts = torch.clamp(query_verts, min=0, max=1)

            out = self.geometry_net(query_verts).to(query_verts.dtype)
            if self.learn_tet_vert_deform_with_mlp:
                self.sdf, self.deform = out[..., 0], out[..., 1:]  # (V,), (V, 3)
            else:
                self.sdf = out.squeeze(-1)  # (V,)
            self.sdf = torch.tanh(self.sdf) * self.scale.mean() * 0.5  # [FIXME] (-1, 1) to (-scale/2, scale/2) or (-scale, scale)?

    def save_init_sdf(self, init_sdf_path, init_sdf, print):
        print(f"Save pre-trained sdf to {init_sdf_path}")
        np.savez_compressed(init_sdf_path, sdf=init_sdf)

    def create_init_sdf(self, mesh, print):
        print(f"pre-trained sdf not found, start training")
        from mesh_to_sdf import mesh_to_sdf

        print("Convert mesh to sdf")
        init_sdf = mesh_to_sdf(mesh, self.verts.cpu().numpy(), sign_method='depth')  # To avoid buble problem, see https://github.com/marian42/mesh_to_sdf#common-parameters
            
        return init_sdf

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

    def generate_edges(self, indices=None):
        # To compute SDF regularization BCE loss
        with torch.no_grad():
            indices = self.indices if indices is None else indices
            edges = torch.tensor([0,1,0,2,0,3,1,2,1,3,2,3], dtype = torch.long, device = "cuda")
            all_edges = indices[:,edges].reshape(-1,2)
            all_edges_sorted = torch.sort(all_edges, dim=1)[0]
            self.all_edges = torch.unique(all_edges_sorted, dim=0)

    @torch.no_grad()
    def getAABB(self):
        return torch.min(self.verts, dim=0).values, torch.max(self.verts, dim=0).values

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
            raise ValueError(f"SMPL skinning weight is used for fixed topology with 6890 points, DMTet change topology.")
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

    def getMesh(self, material, target=None, update_base_mesh=True, update_weights=True, add_non_rigid_offsets=True):
        if update_base_mesh or self.dmtet_verts is None:
            LOGGER.debug("update base mesh in `getMesh`")
            self.update_base_mesh()

        imesh = mesh.Mesh(self.dmtet_verts, self.dmtet_faces, v_tex=self.dmtet_uvs, t_tex_idx=self.dmtet_uv_idx, material=material)

        # Run mesh operations to generate tangent space
        if target is None:
            imesh = mesh.auto_normals(imesh)
            imesh = mesh.compute_tangents(imesh)
            return imesh

        # [FIXME] unchecked code, maybe the transformed normal and tangent are not correct.
        imesh = self.updatePosedMesh(imesh, target, update_weights=update_weights, add_non_rigid_offsets=add_non_rigid_offsets)
        imesh = mesh.batched_auto_normals(imesh)
        imesh = mesh.batched_compute_tangents(imesh)

        # [Note] the mesh
        # jmesh = mesh.Mesh(imesh.v_pos[0], faces, v_tex=uvs, t_tex_idx=uv_idx, material=material)
        # jmesh = mesh.auto_normals(jmesh)
        # jmesh = mesh.compute_tangents(jmesh)

        # print(torch.allclose(imesh.v_nrm[0], jmesh.v_nrm, atol=1e-5), torch.abs(imesh.v_nrm[0]- jmesh.v_nrm).mean(), torch.allclose(imesh.v_tng[0], jmesh.v_tng, atol=1e-5), torch.abs(imesh.v_tng[0]- jmesh.v_tng).mean())

        return imesh

    def prune_tetmesh(self, glctx=None, padding_percent=0.0):
        with torch.no_grad():
            self.update_base_mesh()
            bbox_sdf_func = get_mesh_bbox_sdf(self.dmtet_verts.detach().cpu().numpy(), self.dmtet_faces.detach().cpu().numpy(), padding_percent=padding_percent, nvdiffrast_context=glctx)
            bbox_vert_sdf = bbox_sdf_func(self.tet_verts_deformed.detach().cpu().numpy())
            occ_n = bbox_vert_sdf > 0
            occ_fx4 = occ_n[self.indices.reshape(-1).detach().cpu().numpy()].reshape(-1, 4)
            occ_sum = occ_fx4.sum(axis=-1)
            valid_tets = occ_sum < 4
            valid_tets = torch.from_numpy(valid_tets).to(self.indices.device)
            num_tets = valid_tets.sum().item()

            valid_tets = self.valid_tets | valid_tets  # [NOTE] bbox pruneing may remove more valid tets for the real mesh 
            num_saved_tets = valid_tets.sum().item() - num_tets
            
            valid_indices = self.indices[valid_tets]
            valid_verts_idx, new_indices = torch.unique(valid_indices.view(-1), return_inverse=True)

            new_indices = new_indices.view(-1, 4)
            new_verts = self.verts[valid_verts_idx]
            new_sdf = self.sdf[valid_verts_idx]
            new_deform = self.deform[valid_verts_idx]

        num_verts, num_indices = new_verts.shape[0], new_indices.shape[0]
        old_num_verts, old_num_indices = self.verts.shape[0], self.indices.shape[0]
        print(f"prune tetmesh vertices from {old_num_verts} to {num_verts}")
        print(f"prune tetmesh indices from {old_num_indices} to {num_indices}")
        print(f"saved tets: {num_saved_tets}")

        self.verts = new_verts
        self.indices = new_indices
        self.generate_edges()

        if isinstance(self.sdf, torch.nn.Parameter):
            self.sdf = torch.nn.Parameter(new_sdf.data, requires_grad=True)
            self.register_parameter("sdf", self.sdf)
        else:
            self.sdf.data = new_sdf.data
        if isinstance(self.deform, torch.nn.Parameter):
            self.deform = torch.nn.Parameter(new_deform.data, requires_grad=True)
            self.register_parameter("deform", self.deform)
        else:
            self.deform.data = new_deform.data
        self.update_base_mesh()

    def subdivide_tetmesh(self, return_features=False, permanent_subdivide=False, subdivide_tetmesh_type="full"):
        # [FIXME] no use of `permanent_subdivide` and `return_features` flags
        self.update_base_mesh()  # To update features
        batch_features = self.sdf.unsqueeze(-1)
        if permanent_subdivide:
            with torch.no_grad():
                if subdivide_tetmesh_type == "surface":
                    # Padding surrounding tets to be divided, which removes the holes after subdivision
                    surround_valid_tets = self.valid_tets.clone()
                    for tetrahedron in self.indices[self.valid_tets]:
                        for vertex in tetrahedron:
                            mask = torch.isin(self.indices, vertex).any(axis=-1)
                            mask &= torch.isin(self.indices, tetrahedron[tetrahedron == vertex]).any(axis=-1)
                            mask = torch.nonzero(torch.ravel(mask))
                            surround_valid_tets[mask] = True
                    print(f"add surrounding tets: {(surround_valid_tets ^ self.valid_tets).sum()}")

                    new_verts, new_indices, new_features = self._subdivide_tetmesh(self.verts, self.indices, batch_features, surround_valid_tets)
                elif subdivide_tetmesh_type == "full":
                    new_verts, new_indices, new_features = self._subdivide_tetmesh(self.verts, self.indices, batch_features)
                else:
                    raise ValueError(f"Invalid subdivide_tetmesh_type: {subdivide_tetmesh_type}, should be 'surface' or 'full'")

        else:
            new_verts, new_indices, new_features = self._subdivide_tetmesh(self.verts, self.indices, batch_features, self.valid_tets)
        new_sdf = new_features[..., 0]

        print(f"subdivide tetmesh vertices from {self.verts.shape[0]} to {new_verts.shape[0]}")
        print(f"subdivide tetmesh indices from {self.indices.shape[0]} to {new_indices.shape[0]}")

        if return_features:
            return new_verts, new_indices, new_sdf

        self.verts = new_verts
        self.indices = new_indices
        self.generate_edges()
        # [XXX] the deformation range should be halfed
        self.grid_res = self.grid_res * 2
        print(f"subdivide tetmesh grid_res: {self.grid_res}")

        if isinstance(self.sdf, torch.nn.Parameter):
            self.sdf = torch.nn.Parameter(new_sdf.data.squeeze(-1), requires_grad=True)
            self.register_parameter("sdf", self.sdf)
        else:
            self.sdf.data = new_sdf.data.squeeze(-1)

        # [NOTE] The deformation is scale specific... which means it is irrelavant with SDF, 
        # it is used to cater for the more fine-grained SDF-to-mesh conversion
        print("reset deform to zeros, due to the change of grid_res scale")
        zeros_deform = torch.zeros(self.verts.shape[0], self.deform.shape[-1], dtype=new_verts.dtype, device=new_verts.device)
        if isinstance(self.deform, torch.nn.Parameter):
            self.deform = torch.nn.Parameter(zeros_deform.data, requires_grad=True)
            self.register_parameter("deform", self.deform)
        else:
            self.deform.data = zeros_deform.data
        self.update_base_mesh()

    def _subdivide_tetmesh(self, vertices_nx3, tetrahedrons_fx4, features=None, tetrahedrons_mask_f=None):
        if tetrahedrons_mask_f is not None:
            assert tetrahedrons_mask_f.shape[0] == tetrahedrons_fx4.shape[0]

        touched_tetrahedrons_fx4 = tetrahedrons_fx4 if tetrahedrons_mask_f is None else tetrahedrons_fx4[tetrahedrons_mask_f]
        all_edges = touched_tetrahedrons_fx4[:, self.base_tet_edges].reshape(-1, 2)
        all_edges = self.sort_edges(all_edges)
        unique_edges, idx_map = torch.unique(all_edges, dim=0, return_inverse=True)
        # idx_map = idx_map + vertices_nx3.shape[1]
        idx_map = idx_map + vertices_nx3.shape[0]

        pos_feature = torch.cat([vertices_nx3, features], -1) if (features is not None) else vertices_nx3

        # mid_pos_feature = pos_feature[:, unique_edges.reshape(-1)].reshape(
        #     pos_feature.shape[0], -1, 2, pos_feature.shape[-1]).mean(2)
        # new_pos_feature = torch.cat([pos_feature, mid_pos_feature], 1)
        mid_pos_feature = pos_feature[unique_edges.reshape(-1)].reshape(
            -1, 2, pos_feature.shape[-1]).mean(-2)
        new_pos_feature = torch.cat([pos_feature, mid_pos_feature], 0)
        new_pos, new_features = new_pos_feature[..., :3], new_pos_feature[..., 3:]

        idx_a, idx_b, idx_c, idx_d = torch.split(touched_tetrahedrons_fx4, 1, -1)
        idx_ab, idx_ac, idx_ad, idx_bc, idx_bd, idx_cd = idx_map.reshape(-1, 6).split(1, -1)

        tet_1 = torch.stack([idx_a, idx_ab, idx_ac, idx_ad], dim=1)
        tet_2 = torch.stack([idx_b, idx_bc, idx_ab, idx_bd], dim=1)
        tet_3 = torch.stack([idx_c, idx_ac, idx_bc, idx_cd], dim=1)
        tet_4 = torch.stack([idx_d, idx_ad, idx_cd, idx_bd], dim=1)
        tet_5 = torch.stack([idx_ab, idx_ac, idx_ad, idx_bd], dim=1)
        tet_6 = torch.stack([idx_ab, idx_ac, idx_bd, idx_bc], dim=1)
        tet_7 = torch.stack([idx_cd, idx_ac, idx_bd, idx_ad], dim=1)
        tet_8 = torch.stack([idx_cd, idx_ac, idx_bc, idx_bd], dim=1)
        
        if tetrahedrons_mask_f is not None:
            untouched_tetradedrons_fx4 = tetrahedrons_fx4[~tetrahedrons_mask_f].unsqueeze(-1)
            new_tetrahedrons = torch.cat([tet_1, tet_2, tet_3, tet_4, tet_5, tet_6, tet_7, tet_8, untouched_tetradedrons_fx4], dim=0).squeeze(-1)
        else:
            new_tetrahedrons = torch.cat([tet_1, tet_2, tet_3, tet_4, tet_5, tet_6, tet_7, tet_8], dim=0).squeeze(-1)

        LOGGER.debug(f"number verts: {vertices_nx3.shape[0]} -> {new_pos.shape[0]}")
        LOGGER.debug(f"number tet: {tetrahedrons_fx4.shape[0]} -> {new_tetrahedrons.shape[0]}")
        return (new_pos, new_tetrahedrons) if features is None else (new_pos, new_tetrahedrons, new_features)

    def update_base_mesh(self, tet_verts=None, tet_indices=None, tet_sdf=None, tet_deform=None):
        # [FIXME] the input args are not used yet. no need any more (see `self.subdivide_tetmesh`)
        self.update_sdf_and_deform()

        if tet_indices is not None:
            self.generate_edges(tet_indices)
        # Run DM tet to get a base mesh
        tet_verts = self.verts if tet_verts is None else tet_verts
        tet_indices = self.indices if tet_indices is None else tet_indices
        tet_deform = self.deform if tet_deform is None else tet_deform
        tet_sdf = self.sdf if tet_sdf is None else tet_sdf

        # [NOTE] The deformation is scale specific... which means it is irrelavant with SDF, 
        # it is used to cater for the more fine-grained SDF-to-mesh conversion
        if self.FLAGS.enable_tet_vert_deform is True:
            v_deformed = tet_verts + 2 / (self.grid_res * 2) * torch.tanh(tet_deform)
        else:
            v_deformed = tet_verts
        
        if self.FLAGS.subdivide_aware_marching_tet is True:
            with torch.no_grad():
                occ_n = tet_sdf > 0 # (V), [0, 1]
                occ_fx4 = occ_n[tet_indices.reshape(-1)].reshape(-1,4) # (T, 4), range [0, 1]
                occ_sum = torch.sum(occ_fx4, -1)  # (T), range [0, 1, 2]
                valid_tets = (occ_sum>0) & (occ_sum<4)  # (T_valid), range [True, False]
            self.valid_tets = valid_tets
            v_deformed, tet_indices, tet_sdf = self._subdivide_tetmesh(v_deformed, tet_indices, tet_sdf.unsqueeze(-1), valid_tets)
            tet_sdf = tet_sdf[..., 0]

        # [FIXME] the valid_tets should be compute at the SDF value code, not by `marching_tet`
        verts, faces, uvs, uv_idx, valid_tets, tet_idx = self.marching_tets(v_deformed, tet_sdf, tet_indices)
        self.dmtet_verts = verts
        self.dmtet_faces = faces
        self.dmtet_uvs = uvs
        self.dmtet_uv_idx = uv_idx
        self.tet_verts_deformed = v_deformed
        self.dmtet_tet_idx = tet_idx
        # [XXX] to avoid `valid_tet` repeat computation
        if self.FLAGS.subdivide_aware_marching_tet is False:
            self.valid_tets = valid_tets
        if faces.shape[0] == 0:
            print(f"sdf: {self.sdf.min()}, {self.sdf.max()}")
            raise ValueError("DMTet failed to generate mesh")

    def render(self, glctx, target, lgt, opt_material, bsdf=None, return_kd_grad=True, update_base_mesh=True, update_weights=True, add_non_rigid_offsets=True):
        bsdf = self.FLAGS.bsdf if bsdf is None else bsdf
        LOGGER.debug(f"bsdf: {bsdf}")

        opt_mesh = self.getMesh(opt_material, target, update_base_mesh=update_base_mesh, update_weights=update_weights, add_non_rigid_offsets=add_non_rigid_offsets)

        if 'kd_ks_normal' in opt_material and opt_material['kd_ks_normal'].use_texture_conditional_inputs is True:
            conditional_params = target['params']
            if self.FLAGS.get("use_view_direction", False):
                conditional_params = torch.cat([conditional_params, target["view_direction"]], -1)
            opt_material['kd_ks_normal'].register_conditonal_inputs(conditional_params)

        # [FIXME] ~~the "spp" in dataloader might be fixed due to data caching and preloading~~, use FLAGS.spp instead
        return render.render_mesh(self.FLAGS, glctx, opt_mesh, target['mvp'], target['campos'], lgt, target['resolution'], spp=self.FLAGS.spp, 
                                        msaa=True, background=target['background'], bsdf=bsdf, return_kd_grad=return_kd_grad)


    def tick(self, glctx, target, lgt, opt_material, loss_fn, iteration, lpips_loss_fn, add_non_rigid_offsets):

        # ==============================================================================================
        #  Render optimizable object with identical conditions
        # ==============================================================================================
        buffers = self.render(glctx, target, lgt, opt_material, return_kd_grad=self.return_kd_grad, add_non_rigid_offsets=add_non_rigid_offsets)

        # ==============================================================================================
        #  Compute loss
        # ==============================================================================================
        t_iter = iteration / self.FLAGS.iter  # [0, 1], for reg weighting

        # Image-space loss, split into a coverage component and a color component
        color_ref = target['img']

        joined_mask = color_ref[..., 3:] * buffers['shaded'][..., 3:].detach()
        shaded_rgb = buffers['shaded'][..., 0:3] * joined_mask
        ref_rgb = color_ref[..., 0:3] * color_ref[..., 3:]

        num_rand_img_for_img_loss = self.FLAGS.num_rand_img_for_img_loss
        rand_img_for_loss_idx = torch.randint(0, color_ref.shape[0], (num_rand_img_for_img_loss,))
        shaded_rgb, ref_rgb = shaded_rgb[rand_img_for_loss_idx], ref_rgb[rand_img_for_loss_idx]

        mask_percentage_for_img_loss = self.FLAGS.mask_percentage_for_img_loss
        if mask_percentage_for_img_loss is not None:
            mask_percentage_for_img_loss = 0.75
            joined_mask_idx = torch.where(joined_mask[rand_img_for_loss_idx] > 0)
            num_mask_idx = len(joined_mask_idx[0])
            mask_idx_idx = torch.randperm(num_mask_idx)[:int(num_mask_idx * (1 - mask_percentage_for_img_loss))]
            joined_mask_idx = [_[mask_idx_idx] for _ in joined_mask_idx[:-1]]
            shaded_rgb[joined_mask_idx] = shaded_rgb[joined_mask_idx].detach()

        img_loss = loss_fn(shaded_rgb, ref_rgb) * self.FLAGS.img_loss_weight

        mask_loss = torch.nn.functional.mse_loss(buffers['shaded'][..., 3:], color_ref[..., 3:], reduction='none') * self.FLAGS.mask_loss_weight

        # [XXX] cuda loss ops will change the output shape.
        # e.g., [8, 512, 512, 3] -> [8, 128, 64, 1]
        if self.loss_scaling_buffers is None:
            main_losses = dict(img_loss=img_loss.sum() / (ref_rgb.shape[0] * ref_rgb.shape[1] * ref_rgb.shape[2]), mask_loss=mask_loss.mean())
        else:
            img_loss = img_loss.sum(dim=(1,2,3)) / (ref_rgb.shape[1] * ref_rgb.shape[2])
            mask_loss = mask_loss.mean(dim=(1,2,3))
            main_losses = dict(img_loss=img_loss, mask_loss=mask_loss)

            # update buffer
            batch_size = color_ref.shape[0]
            idx_ls = torch.arange(self.loss_scaling_idx, self.loss_scaling_idx + batch_size) % self.num_loss_scaling_buffers
            self.loss_scaling_idx += batch_size

            if self.loss_scaling_type == 'mask':
                loss_weights = mask_loss.detach()
            elif self.loss_scaling_type == 'image':
                loss_weights = img_loss.detach()
            else:
                loss_weights = sum(main_losses.values()).detach()
            
            self.loss_scaling_buffers[idx_ls] = loss_weights

            # compute weighted mask loss
            loss_buffers_mean = self.loss_scaling_buffers.mean()
            loss_buffers_std = self.loss_scaling_buffers.std()
            loss_weights = torch.clamp(
                torch.tanh(
                    divide_no_nan(loss_buffers_mean - loss_weights, loss_buffers_std)
                ) + 1.0, 
                0.0, 
                1.0
            )

            for _k in main_losses:
                main_losses[_k] = (main_losses[_k] * loss_weights).mean()

        if lpips_loss_fn is not None:
            if self.FLAGS.lpips_in_srgb:
                shaded_rgb, ref_rgb = rgb_to_srgb(shaded_rgb), rgb_to_srgb(ref_rgb)
            _shaded_rgb, _ref_rgb = shaded_rgb * 2 - 1, ref_rgb * 2 - 1
            if self.FLAGS.lpips_loss_weight > 0:
                main_losses["lpips_loss"] = self.FLAGS.lpips_loss_weight * lpips_loss_fn.forward(_shaded_rgb.permute(0, 3, 1, 2), _ref_rgb.permute(0, 3, 1, 2)).mean()

        if self.FLAGS.use_training_tricks:
            sdf_weight = self.FLAGS.sdf_regularizer - (self.FLAGS.sdf_regularizer - 0.01)*min(1.0, 4.0 * t_iter)  # drop from sdf_regularizer=0.02 to 0.01 from 0% to 25% iters
            albedo_weight = self.FLAGS.albedo_regularizer * min(1.0, iteration / 500)
            visibility_weight =  self.FLAGS.visibility_regularizer * min(1.0, iteration / 500)
            init_sdf_weight = self.FLAGS.init_sdf_regularizer * min(1.0, iteration / 500)
            ks_weight = self.FLAGS.ks_regularizer * min(1.0, iteration / 500)
            nrm_weight = self.FLAGS.nrm_regularizer * min(1.0, iteration / 500)
            perturb_nrm_weight = self.FLAGS.perturb_nrm_regularizer * min(1.0, iteration / 500)
        else:
            sdf_weight = self.FLAGS.sdf_regularizer
            albedo_weight = self.FLAGS.albedo_regularizer
            visibility_weight =  self.FLAGS.visibility_regularizer
            init_sdf_weight = self.FLAGS.init_sdf_regularizer
            ks_weight = self.FLAGS.ks_regularizer
            nrm_weight = self.FLAGS.nrm_regularizer
            perturb_nrm_weight = self.FLAGS.perturb_nrm_regularizer
        light_weight = self.FLAGS.light_regularizer
        skinning_weight = self.FLAGS.skinning_regularizer
        offset_weight = self.FLAGS.non_rigid_offset_regularizer
        bone_sdf_weight = self.FLAGS.bone_sdf_regularizer
        bone_offset_weight = self.FLAGS.non_rigid_bone_offset_regularizer
        offset_laplace_weight = self.FLAGS.non_rigid_offset_laplace_regularizer
        vertices_laplace_weight = self.FLAGS.vertices_laplace_regularizer
        tet_deform_weight = self.FLAGS.tet_deform_regularizer
        eikonal_weight = self.FLAGS.eikonal_regularizer
        sdf_mlp_lipschitz_weight = self.FLAGS.sdf_mlp_lipschitz_regularizer
        skin_net_lipschitz_weight = self.FLAGS.skin_net_lipschitz_regularizer
        non_rigid_offset_net_lipschitz_weight = self.FLAGS.non_rigid_offset_net_lipschitz_regularizer
        skin_logits_regularizer = self.FLAGS.skin_logits_regularizer
        skin_logits_regularizer_radius = self.FLAGS.skin_logits_regularizer_radius
        invisible_triangle_sdf_regularizer = self.FLAGS.invisible_triangle_sdf_regularizer
        smpl_surface_skinning_regularizser = self.FLAGS.smpl_surface_skinning_regularizser

        reg_losses = {}

        # SDF regularizer
        if sdf_weight > 0:
            reg_losses["sdf"] = sdf_reg_loss(self.sdf, self.all_edges).mean() * sdf_weight # Dropoff to 0.01

        if invisible_triangle_sdf_regularizer > 0:
            visible_tri_idx = buffers["visible_tri_idx"]
            invisible_tri_idx_mask = torch.ones(self.dmtet_tet_idx.shape[0], dtype=torch.bool, device=self.dmtet_tet_idx.device)
            invisible_tri_idx_mask[visible_tri_idx] = False
            # invisible_tri_idx = torch.nonzero(invisible_tri_idx_mask).squeeze(-1)
            invisible_tri_idx = torch.where(invisible_tri_idx_mask==True)[0]

            invisible_tet = self.dmtet_tet_idx[invisible_tri_idx]
            tet_invisible_vert_idx = self.indices[invisible_tet].ravel().unique()
            tet_invisible_vert_sdf = self.sdf[tet_invisible_vert_idx]

            reg_losses["invisible_sdf"] = invisible_triangle_sdf_regularizer * torch.nn.functional.binary_cross_entropy_with_logits(tet_invisible_vert_sdf, torch.ones_like(tet_invisible_vert_sdf))

        # Albedo (k_d) smoothnesss regularizer
        if self.return_kd_grad:
            reg_losses["albedo_kd_grad"] = torch.mean(buffers['kd_grad'][..., :-1] * buffers['kd_grad'][..., -1:]) * albedo_weight
            reg_losses["ks_grad"] = torch.mean(buffers['ks_grad'][..., :-1] * buffers['ks_grad'][..., -1:]) * ks_weight

        if nrm_weight > 0:
            reg_losses["nrm_grad"] = torch.mean(buffers['nrm_grad'][..., :-1] * buffers['nrm_grad'][..., -1:]) * nrm_weight

        if perturb_nrm_weight > 0 and "perturbed_nrm_grad" in buffers:
            reg_losses["perturbed_nrm_grad"] = torch.mean(buffers['perturbed_nrm_grad'][..., :-1] * buffers['perturbed_nrm_grad'][..., -1:]) * perturb_nrm_weight

        # Visibility regularizer
        if visibility_weight > 0:
            reg_losses["visibility_ao"] = torch.mean(buffers['occlusion'][..., :-1] * buffers['occlusion'][..., -1:]) * visibility_weight

        # Light white balance regularizer
        if light_weight > 0:
            reg_losses["light"] = lgt.regularizer() * light_weight

        # init sdf regularizer
        if self.init_sdf is not None and init_sdf_weight > 0:
            # [BUG] inconsistent tensor shape after tetmseh subdivision
            reg_losses["init_sdf"] = (self.init_sdf - self.sdf).abs().mean() * init_sdf_weight

        if self.regularize_sdf and self.geometry_net is not None and bone_sdf_weight > 0:
            bone_samples = sample_on_bone_head_and_tail(self.bone_heads, self.bone_tails, n_per_bone=self.FLAGS.num_samples_per_bone, range=(0.0, 1.0))
            query_verts = (bone_samples - self.AABB[0]) / (bone_samples - self.AABB[0])
            query_verts = torch.clamp(query_verts, min=0, max=1)
            query_verts = query_verts.reshape(-1, 3)

            out = self.geometry_net(query_verts).to(query_verts.dtype)
            if self.learn_tet_vert_deform_with_mlp:
                sdf = out[..., 0]  # (V,)
            else:
                sdf = out.squeeze(-1)  # (V,)
            reg_losses["bone_sdf"] = torch.maximum(sdf - self.bone_sdf_hinge, torch.zeros_like(out)).mean() * bone_sdf_weight

        # vertices laplace regularizer
        if vertices_laplace_weight > 0:
            reg_losses["vertices_laplace"] = vertices_laplace_weight * regularizer.laplace_regularizer_const(self.dmtet_verts, self.dmtet_faces)  # [NOTE] a big bug of laplace regularizer: both inputs should be mesh's rather the tetmesh's
        
        if tet_deform_weight > 0:
            reg_losses["tet_deform"] = tet_deform_weight * torch.norm(self.deform)

        if self.geometry_net is not None and eikonal_weight > 0:
            reg_losses["eikonal"] = eikonal_weight * self.get_eikonal_loss(pts=self._geometry_net_inputs, sdf=self.sdf)

        if self.geometry_net is not None  and'lipschitz' in self.FLAGS.sdf_mlp_type and sdf_mlp_lipschitz_weight > 0:
            reg_losses["sdf_mlp_lipschitz"] = sdf_mlp_lipschitz_weight * self.geometry_net[1].get_lipschitz_loss()
        
        if 'lipschitz' in self.FLAGS.skin_net_encoding_type and skin_net_lipschitz_weight > 0:
            reg_losses["skin_net_lipschitz"] = skin_net_lipschitz_weight * self.skin_net[1].get_lipschitz_loss()
        
        if 'lipschitz' in self.FLAGS.non_rigid_offset_net_encoding_type and non_rigid_offset_net_lipschitz_weight > 0:
            reg_losses["non_rigid_offset_net_lipschitz"] = non_rigid_offset_net_lipschitz_weight * self.non_rigid_offset_net[1].get_lipschitz_loss()

        # bls weight and offset regularizer
        motion_losses = self.get_motion_losses(target.get('params', None))
        motion_losses["bone_skinning"] *= skinning_weight
        if self.learn_non_rigid_offset:
            motion_losses["bone_offset"] *= bone_offset_weight
            if add_non_rigid_offsets:
                motion_losses['offset'] *= offset_weight
            else:
                motion_losses['offset'] = 0
            if offset_laplace_weight > 0:
                motion_losses["offset_laplace"] = regularizer.batch_laplace_regularizer_const(self.non_rigid_offsets, self.dmtet_faces) * offset_laplace_weight

        if skin_logits_regularizer > 0:
            _, jitter_weights_logits = self.query_weights(self.dmtet_verts + torch.normal(mean=0, std=skin_logits_regularizer_radius, size=self.dmtet_verts.shape, device="cuda"))
            jitter_weights_logits = jitter_weights_logits.to(self.lbs_weights_logits.dtype)
            motion_losses["skin_logits_grad"] = skin_logits_regularizer * torch.sum(torch.abs(jitter_weights_logits - self.lbs_weights_logits)) / self.num_uniq_tfs
        
        if smpl_surface_skinning_regularizser > 0:
            motion_losses["smpl_surface_skinning"] = smpl_surface_skinning_regularizser * self.get_smpl_surface_skinning_loss()

        return main_losses, reg_losses, motion_losses, {"sdf_weight": sdf_weight, "albedo_weight": albedo_weight, "visibility_weight": visibility_weight, "light_weight": light_weight, "skinning_weight": skinning_weight, "bone_offset_weight": bone_offset_weight, "init_sdf_weight": init_sdf_weight, "bone_sdf_weight": bone_sdf_weight, "offset_weight": offset_weight, "offset_laplace_weight": offset_laplace_weight, "vertices_laplace_weight": vertices_laplace_weight, "ks_weight": ks_weight, "nrm_weight": nrm_weight, "perturb_nrm_weight": perturb_nrm_weight, "tet_deform_weight": tet_deform_weight, "eikonal_weight": eikonal_weight, "sdf_mlp_lipschitz_weight": sdf_mlp_lipschitz_weight, "skin_net_lipschitz_weight": skin_net_lipschitz_weight, "non_rigid_offset_net_lipschitz_weight": non_rigid_offset_net_lipschitz_weight, "skin_logits_regularizer": skin_logits_regularizer, "skin_logits_regularizer_radius": skin_logits_regularizer_radius, "lpips_loss_weight": self.FLAGS.lpips_loss_weight, "mask_loss_weight": self.FLAGS.mask_loss_weight, "img_loss_weight": self.FLAGS.img_loss_weight, "lpips_in_srgb": self.FLAGS.lpips_in_srgb, "invisible_triangle_sdf_regularizer": invisible_triangle_sdf_regularizer, "smpl_surface_skinning_regularizser": smpl_surface_skinning_regularizser, "num_rand_img_for_img_loss": num_rand_img_for_img_loss, mask_percentage_for_img_loss: "mask_percentage_for_img_loss"}

    def get_motion_losses(self, params=None):
        losses = {}

        bone_skinning_loss = self.get_bone_skinning_loss()
        losses['bone_skinning'] = bone_skinning_loss

        if self.learn_non_rigid_offset:
            losses["bone_offset"] = self.get_bone_offset_loss(params)
            losses["offset"] = (self.non_rigid_offsets ** 2).mean()

        return losses
    
    def get_bone_offset_loss(self, params=None):
        bone_samples = sample_on_bone_head_and_tail(
            self.bone_heads, self.bone_tails, n_per_bone=self.FLAGS.num_samples_per_bone, range=(0.0, 1.0)
        )  # [n_per_bone, n_bones, 3]
        offsets = self.non_rigid_offset_net(bone_samples.reshape(1, -1, 3), params)
        return (offsets**2).mean()

    def get_bone_skinning_loss(self):
        bone_samples = sample_on_bone_head_and_tail(self.bone_heads, self.bone_tails, n_per_bone=self.FLAGS.num_samples_per_bone, range=(0.1, 0.9))
        bone_samples_shapes = bone_samples.shape
        bone_samples = bone_samples.reshape(-1, bone_samples_shapes[-1])

        # _, weights = self.query_weights(bone_samples)
        # weights = weights.float()
        # weights = torch.functional.F.softmax(weights, dim=-1)
        weights, _ = self.query_weights(bone_samples)
        weights = weights.view(bone_samples_shapes[0], bone_samples_shapes[1], -1)
        weights_gt = torch.nn.functional.one_hot(self.bone2uniq_tf, num_classes=self.num_uniq_tfs).expand_as(weights).to(weights)
        bone_skinning_loss = torch.nn.functional.mse_loss(weights, weights_gt)
        return bone_skinning_loss

    def get_eikonal_loss(self, pts, sdf):
        eikonal_term = torch.autograd.grad(outputs=sdf, inputs=pts,
                                     grad_outputs=torch.ones_like(sdf),
                                     create_graph=True)[0]
        eikonal_loss = ((eikonal_term.norm(dim=-1) - 1) ** 2).mean()

        return eikonal_loss

    def get_smpl_surface_skinning_loss(self):
        samples, sample_face_idxs, w0, w1, w2 = sample_points_from_meshes(self.smpl_mesh_pt3d, 1024)
        pred_weights, _ = self.query_weights(samples)
        face_smpl_lbs_weights = self.smpl_lbs_weights[self.smpl_mesh_pt3d.faces_packed()]
        v0, v1, v2 = face_smpl_lbs_weights[:, 0], face_smpl_lbs_weights[:, 1], face_smpl_lbs_weights[:, 2]
        a = v0[sample_face_idxs]
        b = v1[sample_face_idxs]
        c = v2[sample_face_idxs]
        gt_weights = w0[:, :, None] * a + w1[:, :, None] * b + w2[:, :, None] * c

        return torch.abs(pred_weights - gt_weights).sum(-1).mean()