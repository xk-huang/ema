# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
import time
import json
import logging
import os
import random

import numpy as np
import torch
import nvdiffrast.torch as dr
import xatlas
import hydra
from omegaconf import OmegaConf, ListConfig, DictConfig
import wandb
import lpips
from tqdm import tqdm

# Import data readers / generators
from dataset import DatasetZJUMoCap, DatasetZJUMoCapTAVA
from dataset.samplers import IterationBasedBatchSampler
from dataset.dataset_utils import worker_init_fn

# Import topology / geometry trainers
from geometry.dmtet_dyn import DMTetGeometryDyn
from geometry.dlmesh_dyn import DLMeshDyn
from geometry.vectoradam import VectorAdam

import render.renderutils as ru
from render import obj
from render import material
from render import util
from render import mesh
from render import texture
from render import mlptexture
from render import light
from render import render

from utils import training

RADIUS = 3.0

LOGGER = logging.getLogger(__name__)
print = LOGGER.info

# Enable to debug back-prop anomalies
# torch.autograd.set_detect_anomaly(True)

###############################################################################
# Loss setup
###############################################################################

@torch.no_grad()
def createLoss(FLAGS):
    reduction = 'none'
    if FLAGS.loss == "smape":
        return lambda img, ref: ru.image_loss(img, ref, loss='smape', tonemapper='none', reduction=reduction)
    elif FLAGS.loss == "mse":
        return lambda img, ref: ru.image_loss(img, ref, loss='mse', tonemapper='none', reduction=reduction)
    elif FLAGS.loss == "logl1":
        return lambda img, ref: ru.image_loss(img, ref, loss='l1', tonemapper='log_srgb', reduction=reduction)
    elif FLAGS.loss == "logl2":
        return lambda img, ref: ru.image_loss(img, ref, loss='mse', tonemapper='log_srgb', reduction=reduction)
    elif FLAGS.loss == "relmse":
        return lambda img, ref: ru.image_loss(img, ref, loss='relmse', tonemapper='none', reduction=reduction)
    else:
        assert False

###############################################################################
# Mix background into a dataset image
###############################################################################

@torch.no_grad()
def prepare_batch(target, bg_type='black', add_noise_to_params=False):
    assert len(target['img'].shape) == 4, "Image shape should be [n, h, w, c]"
    if bg_type == 'checker':
        background = torch.tensor(util.checkerboard(target['img'].shape[1:3], 8), dtype=torch.float32, device='cuda')[None, ...]
    elif bg_type == 'black':
        background = torch.zeros(target['img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    elif bg_type == 'white':
        background = torch.ones(target['img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    elif bg_type == 'reference':
        background = target['img'][..., 0:3]
    elif bg_type == 'random':
        background = torch.rand(target['img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    else:
        assert False, "Unknown background type %s" % bg_type

    target['mv'] = target['mv'].cuda()
    target['mvp'] = target['mvp'].cuda()
    target['campos'] = target['campos'].cuda()
    target['img'] = target['img'].cuda()
    target['background'] = background
    target['view_direction'] = target['view_direction'].cuda()

    # target['img'] = torch.cat((torch.lerp(background, target['img'][..., 0:3], target['img'][..., 3:4]), target['img'][..., 3:4]), dim=-1)
    # [NOTE] There should be no soft mask
    background_mask = target['img'][..., 3] == 0
    target['img'][..., 0:3][background_mask] = background[background_mask]

    if 'params' in target:
        if add_noise_to_params is True:
            target['params'] = target['params'] + torch.randn_like(target['params']) * 0.1
        target['params'] = target['params'].cuda()

    return target

###############################################################################
# UV - map geometry & convert to a mesh
###############################################################################

@torch.no_grad()
def xatlas_uvmap(glctx, geometry, mat, FLAGS):
    eval_mesh = geometry.getMesh(mat)
    
    # Create uvs with xatlas
    v_pos = eval_mesh.v_pos.detach().cpu().numpy()
    t_pos_idx = eval_mesh.t_pos_idx.detach().cpu().numpy()
    vmapping, indices, uvs = xatlas.parametrize(v_pos, t_pos_idx)

    # Convert to tensors
    indices_int64 = indices.astype(np.uint64, casting='same_kind').view(np.int64)
    
    uvs = torch.tensor(uvs, dtype=torch.float32, device='cuda')
    faces = torch.tensor(indices_int64, dtype=torch.int64, device='cuda')

    new_mesh = mesh.Mesh(v_tex=uvs, t_tex_idx=faces, base=eval_mesh)

    mask, kd, ks, normal = render.render_uv(glctx, new_mesh, FLAGS.texture_res, eval_mesh.material['kd_ks_normal'])
    
    if FLAGS.layers > 1:
        kd = torch.cat((kd, torch.rand_like(kd[...,0:1])), dim=-1)

    kd_min, kd_max = torch.tensor(FLAGS.kd_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.kd_max, dtype=torch.float32, device='cuda')
    ks_min, ks_max = torch.tensor(FLAGS.ks_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.ks_max, dtype=torch.float32, device='cuda')
    nrm_min, nrm_max = torch.tensor(FLAGS.nrm_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.nrm_max, dtype=torch.float32, device='cuda')

    new_mesh.material = material.Material({
        'bsdf'   : mat['bsdf'],
        'kd'     : texture.Texture2D(kd, min_max=[kd_min, kd_max]),
        'ks'     : texture.Texture2D(ks, min_max=[ks_min, ks_max]),
        'normal' : texture.Texture2D(normal, min_max=[nrm_min, nrm_max])
    })

    return new_mesh

@torch.no_grad()
def mesh_uvmap(glctx, geometry, mat, FLAGS):
    new_mesh = geometry.getMesh(mat)

    mask, kd, ks, normal = render.render_uv(glctx, new_mesh, FLAGS.texture_res, new_mesh.material['kd_ks_normal'])
    
    if FLAGS.layers > 1:
        kd = torch.cat((kd, torch.rand_like(kd[...,0:1])), dim=-1)

    kd_min, kd_max = torch.tensor(FLAGS.kd_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.kd_max, dtype=torch.float32, device='cuda')
    ks_min, ks_max = torch.tensor(FLAGS.ks_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.ks_max, dtype=torch.float32, device='cuda')
    nrm_min, nrm_max = torch.tensor(FLAGS.nrm_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.nrm_max, dtype=torch.float32, device='cuda')

    new_mesh.material = material.Material({
        'bsdf'   : mat['bsdf'],
        'kd'     : texture.Texture2D(kd, min_max=[kd_min, kd_max]),
        'ks'     : texture.Texture2D(ks, min_max=[ks_min, ks_max]),
        'normal' : texture.Texture2D(normal, min_max=[nrm_min, nrm_max])
    })

    return new_mesh

###############################################################################
# Utility functions for material
###############################################################################

def initial_guess_material(geometry, mlp, FLAGS, init_mat=None):
    kd_min, kd_max = torch.tensor(FLAGS.kd_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.kd_max, dtype=torch.float32, device='cuda')
    ks_min, ks_max = torch.tensor(FLAGS.ks_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.ks_max, dtype=torch.float32, device='cuda')
    nrm_min, nrm_max = torch.tensor(FLAGS.nrm_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.nrm_max, dtype=torch.float32, device='cuda')
    if mlp:
        mlp_min = torch.cat((kd_min[0:3], ks_min, nrm_min), dim=0)
        mlp_max = torch.cat((kd_max[0:3], ks_max, nrm_max), dim=0)
        mlp_map_opt = mlptexture.MLPTexture3D(geometry.getAABB(), channels=9, min_max=[mlp_min, mlp_max], internal_dims=FLAGS.mlp_texture_internal_dims, FLAGS=FLAGS)
        mat =  material.Material({'kd_ks_normal' : mlp_map_opt})
    else:
        # Setup Kd (albedo) and Ks (x, roughness, metalness) textures
        texture_res = list(FLAGS.texture_res)
        if FLAGS.random_textures or init_mat is None:
            num_channels = 4 if FLAGS.layers > 1 else 3
            kd_init = torch.rand(size=texture_res + [num_channels], device='cuda') * (kd_max - kd_min)[None, None, 0:num_channels] + kd_min[None, None, 0:num_channels]
            kd_map_opt = texture.create_trainable(kd_init , texture_res, not FLAGS.custom_mip, [kd_min, kd_max])

            ksR = np.random.uniform(size=texture_res + [1], low=0.0, high=0.01)
            ksG = np.random.uniform(size=texture_res + [1], low=ks_min[1].cpu(), high=ks_max[1].cpu())
            ksB = np.random.uniform(size=texture_res + [1], low=ks_min[2].cpu(), high=ks_max[2].cpu())

            ks_map_opt = texture.create_trainable(np.concatenate((ksR, ksG, ksB), axis=2), texture_res, not FLAGS.custom_mip, [ks_min, ks_max])
        else:
            kd_map_opt = texture.create_trainable(init_mat['kd'], texture_res, not FLAGS.custom_mip, [kd_min, kd_max])
            ks_map_opt = texture.create_trainable(init_mat['ks'], texture_res, not FLAGS.custom_mip, [ks_min, ks_max])

        # Setup normal map
        if FLAGS.random_textures or init_mat is None or 'normal' not in init_mat:
            normal_map_opt = texture.create_trainable(np.array([0, 0, 1]), texture_res, not FLAGS.custom_mip, [nrm_min, nrm_max])
        else:
            normal_map_opt = texture.create_trainable(init_mat['normal'], texture_res, not FLAGS.custom_mip, [nrm_min, nrm_max])

        mat = material.Material({
            'kd'     : kd_map_opt,
            'ks'     : ks_map_opt,
            'normal' : normal_map_opt
        })

    if init_mat is not None:
        mat['bsdf'] = init_mat['bsdf']
    else:
        mat['bsdf'] = 'pbr'

    mat["no_perturbed_nrm"] = FLAGS.no_perturbed_nrm
    print(f"no_perturbed_nrm: {FLAGS.no_perturbed_nrm}")

    return mat

###############################################################################
# Validation & testing
###############################################################################

class FunctionTimers:
    def __init__(self):
        import time
        import numpy as np
        from collections import defaultdict

        self.timers = defaultdict(list)

    def get_timer(self, f):
        from functools import wraps
        @wraps(f)
        def _decorate(*args, **kwargs):
            start = time.perf_counter()
            ret = f(*args, **kwargs)
            end = time.perf_counter()
            if f.__name__ not in self.timers:
                self.timers[f.__name__] = []
            self.timers[f.__name__].append((end - start) * 1000)
            return ret

        return _decorate

    def clear(self):
        for k in self.timers:
            self.timers[k] = []

    def report(self):
        return {f"{k}_in_ms": np.mean(v) for k, v in self.timers.items()}


def validate_itr(glctx, target, geometry: DMTetGeometryDyn, opt_material, lgt, FLAGS, update_base_mesh=False, build_mips=True):
    result_dict = {}
    extra_dict = {}
    with torch.no_grad():
        if build_mips:
            lgt.build_mips()
        if FLAGS.camera_space_light:
            lgt.xfm(target['mv'])

        if hasattr(geometry, 'update_base_mesh') and update_base_mesh:
            LOGGER.debug("Updating base mesh in `validate_itr`")
            geometry.update_base_mesh()

        # LOGGER.debug("Updating base mesh in `validate_itr`")
        # geometry.update_base_mesh()  # [NOTE] unconment this line, the the iter per sec of outer loop rushes to 100+; match the FPS metrics

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        # render_start_time = time.time()
        buffers = geometry.render(glctx, target, lgt, opt_material, return_kd_grad=False, update_base_mesh=False, update_weights=False)

        result_dict['ref'] = util.rgb_to_srgb(target['img'][...,0:3])[0]
        result_dict['opt'] = util.rgb_to_srgb(buffers['shaded'][...,0:3])[0]

        result_dict['opt_alpha'] = buffers['shaded'][...,3][0]
        result_dict['ref_alpha'] = target['img'][...,3][0]
        result_image = torch.cat([result_dict['opt'], result_dict['ref']], axis=1)
        resolution = result_dict['ref'].shape[:2]
        end.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()
        # render_time = 1000 * (time.time() - render_start_time)
        render_time = start.elapsed_time(end)
        render_time_per_frame = render_time / target['img'].shape[0]
        extra_dict["render_time_ms_per_frame"] = render_time_per_frame
        extra_dict["fps"] = 1.0 / (render_time_per_frame / 1000.0)

        if FLAGS.display is not None:
            LOGGER.debug(f"Displaying results with displat={FLAGS.display}")
            white_bg = torch.ones_like(target['background'])
            for layer in FLAGS.display:
                if 'latlong' in layer and layer['latlong']:
                    if isinstance(lgt, light.EnvironmentLight):
                        result_dict['light_image'] = util.cubemap_to_latlong(lgt.base, resolution)
                    result_image = torch.cat([result_image, result_dict['light_image']], axis=1)
                elif 'relight' in layer:
                    if not isinstance(layer['relight'], light.EnvironmentLight):
                        layer['relight'] = light.load_env(layer['relight'])
                    img = geometry.render(glctx, target, layer['relight'], opt_material, return_kd_grad=False, update_base_mesh=False, update_weights=False)
                    result_dict['relight'] = util.rgb_to_srgb(img[..., 0:3])[0]
                    result_image = torch.cat([result_image, result_dict['relight']], axis=1)
                elif 'bsdf' in layer:
                    buffers = geometry.render(glctx, target, lgt, opt_material, bsdf=layer['bsdf'], return_kd_grad=False, update_base_mesh=False, update_weights=False)
                    if layer['bsdf'] == 'kd':
                        result_dict[layer['bsdf']] = util.rgb_to_srgb(buffers['shaded'][0, ..., 0:3])  
                    elif layer['bsdf'] == 'normal_camera_space':
                        gb_normal = (buffers['shaded'][0, ..., 0:3] * 2) - 1
                        view_normal = torch.matmul(target["mv"][0,:3,:3][None, None], gb_normal[..., None]).squeeze(-1)
                        view_normal = view_normal / view_normal.norm(dim=-1)[..., None]
                        view_normal = (view_normal + 1) * 0.5
                        lerped_view_normal = torch.lerp(target["background"][0], view_normal, buffers['shaded'][0, ..., 3:4])
                        result_dict[layer['bsdf']] = lerped_view_normal[..., 0:3]
                    else:
                        result_dict[layer['bsdf']] = buffers['shaded'][0, ..., 0:3]
                    result_image = torch.cat([result_image, result_dict[layer['bsdf']]], axis=1)
   
        return result_image, result_dict, extra_dict

def validate(glctx, geometry, opt_material, lgt, dataset_validate, out_dir, FLAGS, save_gts=True):
    """_summary_

    Args:
        glctx (_type_): _description_
        geometry (_type_): _description_
        opt_material (_type_): _description_
        lgt (_type_): _description_
        dataset_validate (_type_): _description_
        out_dir (_type_): _description_
        FLAGS (_type_): _description_

    Returns:
        Dict[str, np.ndarray]: _description_
    """
    # ==============================================================================================
    #  Validation loop
    # ==============================================================================================
    mse_values = []
    psnr_values = []
    ssim_values = []
    fps_values = []

    LOGGER.warning("In `validate`, the batch size is foreced to be 1")  # [XXX]
    dataloader_validate = torch.utils.data.DataLoader(dataset_validate, batch_size=1, collate_fn=dataset_validate.collate, worker_init_fn=worker_init_fn)

    if hasattr(geometry, 'update_base_mesh'):
        LOGGER.debug("Updating base mesh in `validate`")
        geometry.update_base_mesh()

    compute_val_metrics_with_mask = FLAGS.compute_val_metrics_with_mask
    print(f"compute_val_metrics_with_mask={compute_val_metrics_with_mask}")

    function_timers = None
    if FLAGS.get("runtime_breakdown", False) is True:
        function_timers = FunctionTimers()
        geometry.getMesh = function_timers.get_timer(geometry.getMesh)
        geometry.update_base_mesh = function_timers.get_timer(geometry.update_base_mesh)
        render.render_mesh = function_timers.get_timer(render.render_mesh)
        render.render_layer = function_timers.get_timer(render.render_layer)
        render.shade = function_timers.get_timer(render.shade)

    validate_update_base_mesh = FLAGS.get("validate_update_base_mesh", False)
    print(f"validate_update_base_mesh: {validate_update_base_mesh}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'metrics.txt'), 'w') as fout:
        fout.write('ID, MSE, PSNR, SSIM, FPS\n')

        print(f"Running validation @ {out_dir}")
        for it, target in enumerate(tqdm(dataloader_validate)):

            # Mix validation background
            target = prepare_batch(target, FLAGS.background)

            result_image, result_dict, extra_dict = validate_itr(glctx, target, geometry, opt_material, lgt, FLAGS, update_base_mesh=validate_update_base_mesh, build_mips=(True if it == 0 else False))
            fps_values.append(extra_dict['fps'])

            if FLAGS.compute_val_metrics is True:
                # Compute metrics
                opt = torch.clamp(result_dict['opt'], 0.0, 1.0) 
                ref = torch.clamp(result_dict['ref'], 0.0, 1.0)

                if compute_val_metrics_with_mask is True:
                    mask_at_box = target["mask_at_box"][0]
                    result_dict["mask_at_box"] = mask_at_box * 1.0

                    mse = torch.nn.functional.mse_loss(opt[mask_at_box], ref[mask_at_box], size_average=None, reduce=None, reduction='mean').item()
                    psnr = util.mse_to_psnr(mse)
                    ssim = util.compute_ssim(opt, ref, mask_at_box)
                else:
                    mse = torch.nn.functional.mse_loss(opt, ref, size_average=None, reduce=None, reduction='mean').item()
                    psnr = util.mse_to_psnr(mse)
                    ssim = util.compute_ssim(opt, ref)

                mse_values.append(float(mse))
                psnr_values.append(float(psnr))
                ssim_values.append(float(ssim))

                line = "%d, %1.8f, %1.8f, %1.8f, %.2f\n" % (it, mse, psnr, ssim, fps_values[-1])
                fout.write(str(line))

            if FLAGS.save_val_images is True:
                save_val_keys = set(result_dict.keys())
                if not save_gts:
                    save_val_keys.remove('ref')
                    save_val_keys.remove('ref_alpha')
                    if compute_val_metrics_with_mask is True:
                        save_val_keys.remove('mask_at_box')

                # make directories before saving
                for k in save_val_keys:
                    sub_dir = os.path.join(out_dir, k)
                    if not os.path.exists(sub_dir):
                        os.makedirs(sub_dir, exist_ok=True)

                for k in save_val_keys:
                    # [NOTE] every light image is the same, we only save one
                    if k == 'light_image' and it > 0:
                        continue
                    np_img = result_dict[k].detach().cpu().numpy()
                    sub_dir = os.path.join(out_dir, k)
                    util.save_image(os.path.join(sub_dir, f'val-{it:06d}-{target["frame_id"][0]}-{target["camera_id"][0]}.png'), np_img)

        avg_mse = np.mean(np.array(mse_values))
        avg_psnr = np.mean(np.array(psnr_values))
        avg_ssim = np.mean(np.array(ssim_values))
        avg_fps = np.mean(np.array(fps_values))
        line = "AVERAGES: %1.4f, %2.6f, %2.6f, %.6f\n" % (avg_mse, avg_psnr, avg_ssim, avg_fps)
        fout.write(str(line))
        print("MSE,      PSNR,    SSIM,    FPS")
        print("%1.8f, %2.6f, %2.6f, %.6f" % (avg_mse, avg_psnr, avg_ssim, avg_fps))

    runtime_breakdown_report = function_timers.report() if function_timers is not None else {}
    print(f"Runtime breakdown report: {runtime_breakdown_report}")
    return {
        'mse': avg_mse,
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'fps': avg_fps
    }

###############################################################################
# Main shape fitter function / optimization loop
###############################################################################

class Trainer(torch.nn.Module):
    def __init__(self, glctx, geometry, lgt, mat, optimize_geometry, optimize_light, image_loss_fn, FLAGS, lpips_loss_fn):
        super(Trainer, self).__init__()

        self.glctx = glctx
        self.geometry = geometry
        self.light = lgt
        self.material = mat
        self.optimize_geometry = optimize_geometry
        self.optimize_light = optimize_light
        self.image_loss_fn = image_loss_fn
        self.FLAGS = FLAGS
        self.lpips_loss_fn = lpips_loss_fn

        if not self.optimize_light:
            with torch.no_grad():
                self.light.build_mips()

        self.update_params()

        if self.geometry.geometry_net is not None and "windowed" in self.FLAGS.sdf_mlp_type:
            if self.FLAGS.sdf_mlp_postional_encoding_anneal_steps <= 0:
                raise ValueError(f"Geometry Net Positional encoding annealing steps must be > 0, got {self.FLAGS.sdf_mlp_postional_encoding_anneal_steps}")
            self.sdf_mlp_postional_encoding_anneal_base = self.FLAGS.sdf_mlp_num_freq / self.FLAGS.sdf_mlp_postional_encoding_anneal_steps

            print(f"Geometry Net Positional encoding annealing base: {self.sdf_mlp_postional_encoding_anneal_base} = {self.FLAGS.sdf_mlp_num_freq} / {self.FLAGS.sdf_mlp_postional_encoding_anneal_steps}")
        else:
            self.sdf_mlp_postional_encoding_anneal_base = None
        
        if self.geometry.skin_net is not None and "windowed" in self.FLAGS.skin_net_encoding_type:
            if self.FLAGS.skin_net_postional_encoding_anneal_steps <= 0:
                raise ValueError(f"Skin Net Positional encoding annealing steps must be > 0, got {self.FLAGS.skin_net_postional_encoding_anneal_steps}")

            self.skin_net_postional_encoding_anneal_base = self.FLAGS.skin_net_num_freq / self.FLAGS.skin_net_postional_encoding_anneal_steps
            print(f"Skin Net Positional encoding annealing base: {self.skin_net_postional_encoding_anneal_base} = {self.FLAGS.skin_net_num_freq} / {self.FLAGS.skin_net_postional_encoding_anneal_steps}")
        else:
            self.skin_net_postional_encoding_anneal_base = None
        
        if self.geometry.non_rigid_offset_net is not None and "windowed" in self.FLAGS.non_rigid_offset_net_encoding_type:
            if self.FLAGS.non_rigid_offset_net_postional_encoding_anneal_steps <= 0:
                raise ValueError(f"Non Rigid Net Positional encoding annealing steps must be > 0, got {self.FLAGS.non_rigid_offset_net_postional_encoding_anneal_steps}")

            self.non_rigid_offset_net_postional_encoding_anneal_base = self.FLAGS.non_rigid_offset_net_num_freq / self.FLAGS.non_rigid_offset_net_postional_encoding_anneal_steps
            print(f"Non Rigid Net Positional encoding annealing base: {self.non_rigid_offset_net_postional_encoding_anneal_base} = {self.FLAGS.non_rigid_offset_net_num_freq} / {self.FLAGS.non_rigid_offset_net_postional_encoding_anneal_steps}")
        else:
            self.non_rigid_offset_net_postional_encoding_anneal_base = None

        if FLAGS.no_perturbed_nrm is False:
            self.perturbed_nrm_start_iter = FLAGS.perturbed_nrm_start_iter
            print(f"Pertueb nrm, start_iter: {self.perturbed_nrm_start_iter}")
        else:
            self.perturbed_nrm_start_iter = np.inf
            print("NOT pertueb nrm")

        print("Build Trainer")

    def update_params(self):
        geometry = self.geometry
        optimize_geometry = self.optimize_geometry, 
        optimize_light = self.optimize_light, 
        FLAGS = self.FLAGS
        self.params = list(self.material.parameters())
        self.params += list(self.light.parameters()) if optimize_light else []
        self.params = [{"params": self.params, "initial_lr": FLAGS.learning_rate_material}]
        self.geo_params = [{"params": params, "initial_lr": FLAGS.learning_rate_geometry} for name, params in geometry.named_parameters() if not name.startswith('skin_net')] if optimize_geometry else []
        self.motion_params = [{"params": params, "initial_lr": FLAGS.learning_rate_motion} for params in geometry.skin_net.parameters()] if geometry.skin_net is not None else []
        assert len(self.geo_params) + len(self.motion_params) == len(list(geometry.parameters())), f"Geometry parameters not fully optimized: {len(self.geo_params)} + {len(self.motion_params)} != {len(list(geometry.parameters()))}"

        print(f"Material & light parameters: {len(self.params)}")
        print(f"Geometry parameters: {len(self.geo_params)}")
        print(f"Motion parameters: {len(self.motion_params)}")


    def forward(self, target, it):
        if self.optimize_light:
            self.light.build_mips()
            if self.FLAGS.camera_space_light:
                self.light.xfm(target['mv'])

        # Split non rigid net offsets
        add_non_rigid_offsets = (it >= self.FLAGS.split_non_rigid_offset_net_optim_steps)

        if self.sdf_mlp_postional_encoding_anneal_base is not None:
            LOGGER.debug(f"Geometry Net Positional encoding annealing alpha: {it * self.sdf_mlp_postional_encoding_anneal_base}")
            self.geometry.geometry_net[0].register_alpha(it * self.sdf_mlp_postional_encoding_anneal_base)

        if self.skin_net_postional_encoding_anneal_base is not None:
            LOGGER.debug(f"Skin Net Positional encoding annealing alpha: {it * self.skin_net_postional_encoding_anneal_base}")
            self.geometry.skin_net[0].register_alpha(it * self.skin_net_postional_encoding_anneal_base)

        if self.non_rigid_offset_net_postional_encoding_anneal_base is not None:
            LOGGER.debug(f"Non Rigid Net Positional encoding annealing alpha: {(it - self.FLAGS.split_non_rigid_offset_net_optim_steps) * self.non_rigid_offset_net_postional_encoding_anneal_base}")
            self.geometry.non_rigid_offset_net[0].register_alpha((it - self.FLAGS.split_non_rigid_offset_net_optim_steps) * self.non_rigid_offset_net_postional_encoding_anneal_base)

        self.material["no_perturbed_nrm"] = (it < self.perturbed_nrm_start_iter)

        return self.geometry.tick(self.glctx, target, self.light, self.material, self.image_loss_fn, it, self.lpips_loss_fn, add_non_rigid_offsets)

def optimize_mesh(
    glctx,
    geometry,
    opt_material,
    lgt,
    dataset_train,
    dataset_validate,
    FLAGS,
    warmup_iter=0,
    log_interval=10,
    pass_idx=0,
    pass_name="",
    optimize_light=True,
    optimize_geometry=True
    ):

    # ==============================================================================================
    #  Setup torch optimizer
    # ==============================================================================================

    learning_rate_pos = FLAGS.learning_rate_geometry
    learning_rate_mat = FLAGS.learning_rate_material
    learning_rate_skn = FLAGS.learning_rate_motion

    # if FLAGS.use_training_tricks is True:
    def lr_schedule(
        iter, 
        fraction=(1/(FLAGS.iter - warmup_iter)), 
        learning_rate_final_mult=np.log10(1/FLAGS.learning_rate_final_mult) if FLAGS.learning_rate_schedule_type == 'exponential' else FLAGS.learning_rate_final_mult,
        re_warmup_iters=FLAGS.subdivide_tetmesh_iters, 
        re_warmup_interval=FLAGS.subdivide_learning_rate_warmup_interval, 
        re_warmup_lr_mult=FLAGS.subdivide_learning_rate_mult, 
        learning_rate_schedule_type=FLAGS.learning_rate_schedule_type,
        learning_rate_schedule_steps=FLAGS.learning_rate_schedule_steps,
        ):
        if iter < warmup_iter:
            adjust_mult = iter / warmup_iter 
        elif learning_rate_schedule_type == "exponential":
            # Exponential falloff from [1.0, learning_rate_final_mult] over (iter - warmup_iter) epochs.    
            adjust_mult = max(0.0, 10**(-(iter - warmup_iter)*fraction*learning_rate_final_mult)) 
        elif learning_rate_schedule_type == "linear":
            adjust_mult = max(0.0, 1.0-(iter - warmup_iter)*fraction*(1.0-learning_rate_final_mult))
        elif learning_rate_schedule_type == "cosine":
            adjust_mult = max(0.0, learning_rate_final_mult+0.5*(1.0 - learning_rate_final_mult)*(1.0+np.cos(np.pi*(iter - warmup_iter)*fraction)))
        elif learning_rate_schedule_type == "step":
            adjust_mult = max(0.0, learning_rate_final_mult**(sum([_step <= iter for _step in learning_rate_schedule_steps])))
        for re_warmup_iter in re_warmup_iters:
            if iter >= re_warmup_iter and iter < re_warmup_iter + re_warmup_interval:
                adjust_mult *= max((iter - re_warmup_iter) / re_warmup_interval, re_warmup_lr_mult)
        return adjust_mult
    # print(f"use_training_tricks is True, enabling lr_schedule nd loss weight schedule")
    print(f"Exponential falloff of lr from [lr * 1.0, lr * {FLAGS.learning_rate_final_mult}] from {warmup_iter} to {FLAGS.iter}")
    # elif FLAGS.use_training_tricks is False:
    #     def lr_schedule(_):
    #         return 1.0
    #     print(f"use_training_tricks is False, disabling lr_schedule and loss weight schedule")
    # else:
    #     raise ValueError(f"Invalid value for FLAGS.use_training_tricks: {FLAGS.use_training_tricks}")

    # ==============================================================================================
    #  Image loss
    # ==============================================================================================
    image_loss_fn = createLoss(FLAGS)

    if FLAGS.lpips_loss_weight > 0:
        print("Loading LPIPS model")
        lpips_loss_fn = lpips.LPIPS(net='vgg').cuda()
    else:
        lpips_loss_fn = None

    trainer_noddp = Trainer(glctx, geometry, lgt, opt_material, optimize_geometry, optimize_light, image_loss_fn, FLAGS, lpips_loss_fn)

    # Relighting
    if FLAGS.no_train is True and FLAGS.learn_light is False:
            print(f"Do not load lights from checkpoint, using loaded lights {FLAGS.envmap}")
            trainer_noddp.light = None
    
    if FLAGS.external_ckpt_dir is not None:
        # try to load weights if the sdf or deform weights are mis-matched
        try:
            training.resume_from_ckpt(FLAGS.external_ckpt_dir, trainer_noddp, optimizer=None, optimizer_mesh=None, optimizer_motion=None, step=FLAGS.external_ckpt_step)
        except RuntimeWarning as e:
            print("Fail to load weights due to shape mismatch. Copy the mismatched weights and retrying...")
            training.resume_from_ckpt(FLAGS.external_ckpt_dir, trainer_noddp, optimizer=None, optimizer_mesh=None, optimizer_motion=None, step=FLAGS.external_ckpt_step, strict=True)

    # No training if validate_only is True
    if FLAGS.no_train is True:
        print(f"Rendering only, skipping training.")
        return geometry, opt_material

    # build optimizer
    trainer, optimizer_mesh, optimizer_motion, optimizer = build_optims(trainer_noddp, FLAGS, optimize_geometry, learning_rate_pos, learning_rate_mat, learning_rate_skn)

    if FLAGS.external_ckpt_dir is not None:
        # [FIXME] last_step should be used when initializing
        last_step = training.resume_from_ckpt(FLAGS.external_ckpt_dir, trainer_noddp, optimizer, optimizer_mesh, optimizer_motion, FLAGS.external_ckpt_step)
        print(f"Loading external checkpoint from {FLAGS.external_ckpt_dir}, last step is {last_step}")
    else:
        last_step = -1

    # ==============================================================================================
    # Schedular
    # ==============================================================================================
    scheduler, scheduler_mesh, scheduler_motion = build_scheds(FLAGS, lr_schedule, optimizer_mesh, optimizer_motion, optimizer, last_step)

    # ==============================================================================================
    #  Training loop
    # ==============================================================================================
    img_cnt = 0
    main_loss_vec = []
    reg_loss_vec = []
    motion_loss_vec = []
    iter_opt_dur_vec = []
    iter_data_dur_vec = []

    if FLAGS.multi_gpu:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset_train, shuffle=True, num_replicas=FLAGS.world_size, rank=FLAGS.rank)
    else:
        sampler = torch.utils.data.sampler.RandomSampler(dataset_train)

    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, FLAGS.batch, drop_last=False)
    num_iters_per_epoch = len(batch_sampler)
    # [NOTE] FLAGS.iter + 1, to save and eval the model at the last step
    # [XXX] in fact, we run one more extra step...
    iter_based_batch_sampler = IterationBasedBatchSampler(batch_sampler, FLAGS.iter + 1, last_step + 1)

    dataloader_train = torch.utils.data.DataLoader(dataset_train, collate_fn=dataset_train.collate, num_workers=FLAGS.num_workers_train, pin_memory=True, batch_sampler=iter_based_batch_sampler, worker_init_fn=worker_init_fn)

    if FLAGS.val_batch != 1:
        raise ValueError(
            f"val_batch must be 1 to properly write images, got {FLAGS.val_batch}."
            "`validate_itr` only return the first image of the batch."
        )
    LOGGER.warning(f"val_batch is not used in `validate`. In `validate` the batch size is foreced to be 1.")

    if not isinstance(dataset_validate, list):
        dataset_validate = [dataset_validate]

    dataloader_validate = [torch.utils.data.DataLoader(dataset_validate_, batch_size=FLAGS.val_batch, collate_fn=dataset_train.collate, num_workers=FLAGS.num_workers_validate, pin_memory=True, worker_init_fn=worker_init_fn) for dataset_validate_ in dataset_validate]
    
    if FLAGS.num_workers_validate != 0:
        LOGGER.warning(f"num_workers_validate is not 0, this may cause a block when using multiple dataloader_validate.")

    def cycle(iterable):
        cnt = 0
        iterator = iter(iterable)
        while True:
            try:
                print(f"cycle iterable cnt: {cnt}")
                yield next(iterator)
                cnt += 1
            except StopIteration:
                print("Restarting iterator")
                iterator = iter(iterable)

    v_it = [cycle(dataloader_validate_) for dataloader_validate_ in dataloader_validate]

    print(f"Training {pass_name} pass with {len(dataloader_train)} batches per epoch. Each batch has {FLAGS.batch} images.")

    # NOTE: Make dirs for saving images and checkpoints
    save_dir = os.path.join(FLAGS.out_dir, "val_vis_in_train")
    os.makedirs(save_dir, exist_ok=True)
    ckpt_dir = os.path.join(FLAGS.out_dir, "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)
    mesh_dir = os.path.join(FLAGS.out_dir, "meshes")
    os.makedirs(mesh_dir, exist_ok=True)
    outlier_ckpt_dir = os.path.join(FLAGS.out_dir, "outlier_ckpts")
    os.makedirs(outlier_ckpt_dir, exist_ok=True)
    outlier_mesh_dir = os.path.join(FLAGS.out_dir, "outlier_meshes")
    os.makedirs(outlier_mesh_dir, exist_ok=True)

    validate_split_names = [dataset_validate_.split for dataset_validate_ in dataset_validate]
    for validate_split_name in validate_split_names:
        os.makedirs(os.path.join(save_dir, validate_split_name), exist_ok=True)

    # log the previous metrics. If current metrics are much worse, then we should check the model weight.
    if FLAGS.monitor_metric_name is not None:
        monitor_metric_name = FLAGS.monitor_metric_name
        monitor_metric_outlier_threshold = FLAGS.monitor_metric_outlier_threshold
        assert monitor_metric_outlier_threshold >= 0

        if FLAGS.monitor_metric_type == "max":
            outlier_metric_previous_value = -np.inf
            better_metric_previous_value = -np.inf
            metric_compare_func = np.greater
        elif FLAGS.monitor_metric_type == "min":
            outlier_metric_previous_value = np.inf
            better_metric_previous_value = np.inf
            metric_compare_func = np.less
            monitor_metric_outlier_threshold = -monitor_metric_outlier_threshold
        else:
            raise ValueError(f"Unknown monitor_metric_type {FLAGS.monitor_metric_type} from {FLAGS.monitor_metric_name}")
    else:
        monitor_metric_name = None

    # NOTE: Global state variables
    save_out_base_dir_with_gt = None
    iter_data_start_time = time.time()

    print(f"output directory: {FLAGS.out_dir}")
    for it, target in enumerate(dataloader_train):
        # Mix randomized background into dataset image
        target = prepare_batch(target, 'random', add_noise_to_params=FLAGS.add_noise_to_params)
        iter_data_dur_vec.append(time.time() - iter_data_start_time)

        it += last_step + 1
        if it % num_iters_per_epoch == 0:
            epoch = it // num_iters_per_epoch
            print(f"Epoch {epoch} starts at step {it}, every {num_iters_per_epoch} steps.")
            if FLAGS.multi_gpu:
                sampler.set_epoch(epoch)

        if it in FLAGS.prune_tetmesh_iters:
            num_repeat = FLAGS.prune_tetmesh_iters.index(it)
            padding_percent = FLAGS.prune_tetmesh_padding_percent / (2 ** num_repeat)
            print(f"Prune tetmesh at iteration {it}; padding_percent={padding_percent}")
            trainer_noddp.geometry.prune_tetmesh(padding_percent=padding_percent, glctx=glctx)

        if it in FLAGS.subdivide_tetmesh_iters:
            print(f"Subdividing tetmesh at iteration {it}")
            trainer_noddp.geometry.subdivide_tetmesh(permanent_subdivide=True)

            if FLAGS.learn_sdf_with_mlp is False or (FLAGS.learn_tet_vert_deform_with_mlp is False and FLAGS.enable_tet_vert_deform is True):
                print("Rebuild optimizers and schedulers")
                trainer_noddp.update_params()
                _, optimizer_mesh, _, _ = build_optims(trainer_noddp, FLAGS, optimize_geometry, learning_rate_pos, None, None)

                _, scheduler_mesh, _ = build_scheds(FLAGS, lr_schedule, optimizer_mesh, None, None, it - 1)

            if FLAGS.force_reinit_optims_after_subdivide_tetmesh is True:
                print("Rebuild optimizers and schedulers")
                trainer_noddp.update_params()
                _, optimizer_mesh, optimizer_motion, optimizer = build_optims(trainer_noddp, FLAGS, optimize_geometry, learning_rate_pos, learning_rate_mat, learning_rate_skn)

                scheduler, scheduler_mesh, scheduler_motion = build_scheds(FLAGS, lr_schedule, optimizer_mesh, optimizer_motion, optimizer, it - 1)

        # ==============================================================================================
        #  Display / save outputs. Do it before training so we get initial meshes
        # ==============================================================================================

        # Show/save image before training step (want to get correct rendering of input)
        if FLAGS.local_rank == 0:
            display_image = FLAGS.display_interval and (it % FLAGS.display_interval == 0)
            save_image = FLAGS.save_interval and (it % FLAGS.save_interval == 0)
            log_val = FLAGS.log_val_interval and (it % FLAGS.log_val_interval == 0)
            if display_image or save_image or log_val:
                if log_val:
                    metric_log_dict = {}
                for i_v_it, v_it_ in enumerate(v_it):
                    _target = prepare_batch(next(v_it_), FLAGS.background)
                    _mask_at_box = _target["mask_at_box"][0]
                    result_image, result_dict, extra_dict = validate_itr(glctx, _target, geometry, opt_material, lgt, FLAGS, update_base_mesh=True)

                    if display_image:
                        np_result_image = result_image.detach().cpu().numpy()
                        util.display_image(np_result_image, title='%d / %d' % (it, FLAGS.iter))
                    if save_image:
                        np_result_image = result_image.detach().cpu().numpy()
                        util.save_image(os.path.join(save_dir, validate_split_names[i_v_it], ('img_%s_%06d.png' % (pass_name, img_cnt))), np_result_image)

                    if log_val:
                        # Compute metrics
                        val_img_loss = torch.nn.functional.mse_loss(result_dict['opt'], result_dict['ref'])
                        val_alpha_loss = torch.nn.functional.mse_loss(result_dict['opt_alpha'], result_dict['ref_alpha'])

                        # Compute metrics
                        opt = torch.clamp(result_dict['opt'], 0.0, 1.0) 
                        ref = torch.clamp(result_dict['ref'], 0.0, 1.0)

                        if FLAGS.compute_val_metrics_with_mask is True:
                            opt = opt[_mask_at_box]
                            ref = ref[_mask_at_box]

                        mse = torch.nn.functional.mse_loss(opt, ref, size_average=None, reduce=None, reduction='mean').item()
                        val_img_psnr = util.mse_to_psnr(mse)

                        opt_alpha = torch.clamp(result_dict['opt_alpha'], 0.0, 1.0) 
                        ref_alpha = torch.clamp(result_dict['ref_alpha'], 0.0, 1.0)

                        if FLAGS.compute_val_metrics_with_mask is True:
                            opt_alpha = opt_alpha[_mask_at_box]
                            ref_alpha = ref_alpha[_mask_at_box]

                        alpha_mse = torch.nn.functional.mse_loss(opt_alpha, ref_alpha, size_average=None, reduce=None, reduction='mean').item()
                        val_alpha_psnr = util.mse_to_psnr(alpha_mse)

                        metric_log_dict.update({f"val/{validate_split_names[i_v_it]}/val_img_psnr": val_img_psnr, f"val/{validate_split_names[i_v_it]}/val_alpha_pnsr": val_alpha_psnr, f"val/{validate_split_names[i_v_it]}/val_img_loss": val_img_loss.item(), f"val/{validate_split_names[i_v_it]}/val_alpha_loss": val_alpha_loss})
                        metric_log_dict.update({f"time-val/{validate_split_names[i_v_it]}/render_time_ms_per_frame": extra_dict["render_time_ms_per_frame"], f"time-val/{validate_split_names[i_v_it]}/render_fps": extra_dict["fps"]})

                        print(f"iter={it:5d}, split={validate_split_names[i_v_it]}, val_img_psnr={val_img_psnr:.3f}, val_alpha_psnr={val_alpha_psnr:.3f}, render_time_ms_per_frame={extra_dict['render_time_ms_per_frame']:.1f}, render_fps={extra_dict['fps']:.2f}")
                if log_val:
                    wandb.log(metric_log_dict, step=it)

            if save_image:
                img_cnt = img_cnt+1


        iter_start_time = time.time()

        # ==============================================================================================
        #  Zero gradients
        # ==============================================================================================
        optimizer.zero_grad()
        if optimize_geometry:
            optimizer_mesh.zero_grad()
        if optimizer_motion is not None:
            optimizer_motion.zero_grad()

        # ==============================================================================================
        #  Training
        # ==============================================================================================
        main_losses, reg_losses, motion_losses, loss_weights = trainer(target, it)
        main_loss = sum(main_losses.values())
        reg_loss = sum(reg_losses.values())
        motion_loss = sum(motion_losses.values())

        if not isinstance(reg_loss, torch.Tensor):
            reg_loss = torch.tensor(reg_loss)
        if not isinstance(motion_loss, torch.Tensor):
            motion_loss = torch.tensor(motion_loss)
        # ==============================================================================================
        #  Final loss
        # ==============================================================================================
        total_loss = main_loss + reg_loss + motion_loss

        if torch.is_anomaly_enabled():
            assert torch.all(torch.isfinite(total_loss))
            assert torch.all(torch.isfinite(main_loss))
            
            for k, v in reg_losses.items():
                assert torch.all(torch.isfinite(v)), f"{k} {v}"
            
            for k, v in motion_losses.items():
                assert torch.all(torch.isfinite(v)), f"{k} {v}"

        main_loss_vec.append(main_loss.item())
        reg_loss_vec.append(reg_loss.item())
        motion_loss_vec.append(motion_loss.item())

        # ==============================================================================================
        #  Backpropagate
        # ==============================================================================================
        total_loss.backward()

        if FLAGS.use_training_tricks is True:
            if hasattr(lgt, 'base') and lgt.base.grad is not None and optimize_light:
                lgt.base.grad *= 64
            if 'kd_ks_normal' in opt_material:
                opt_material['kd_ks_normal'].encoder.params.grad /= 8.0

        optimizer.step()
        scheduler.step()

        is_iter_to_stop_optim_sdf_mlp = (FLAGS.iter_to_stop_optim_sdf_mlp is not None and it > FLAGS.iter_to_stop_optim_sdf_mlp)
        if optimize_geometry and not is_iter_to_stop_optim_sdf_mlp:
            optimizer_mesh.step()
            scheduler_mesh.step()

        no_motion_update = any([(it >= i and it < i + FLAGS.subdivide_no_motion_update_interval) for i in FLAGS.subdivide_tetmesh_iters])
        if optimizer_motion is not None:
            if not no_motion_update:
                optimizer_motion.step()
            scheduler_motion.step()

        # ==============================================================================================
        #  Clamp trainables to reasonable range
        # ==============================================================================================
        with torch.no_grad():
            if 'kd' in opt_material:
                opt_material['kd'].clamp_()
            if 'ks' in opt_material:
                opt_material['ks'].clamp_()
            if 'normal' in opt_material:
                opt_material['normal'].clamp_()
                opt_material['normal'].normalize_()
            if lgt is not None:
                lgt.clamp_(min=0.0)

        torch.cuda.current_stream().synchronize()
        iter_opt_dur_vec.append(time.time() - iter_start_time)

        # ==============================================================================================
        #  Logging
        # ==============================================================================================
        if it % log_interval == 0 and FLAGS.local_rank == 0:
            main_loss_avg = np.mean(np.asarray(main_loss_vec[-log_interval:]))
            reg_loss_avg = np.mean(np.asarray(reg_loss_vec[-log_interval:]))
            motion_loss_avg = np.mean(np.asarray(motion_loss_vec[-log_interval:]))

            iter_opt_dur_avg = np.mean(np.asarray(iter_opt_dur_vec[-log_interval:]))
            iter_data_dur_avg = np.mean(np.asarray(iter_data_dur_vec[-log_interval:]))
            iter_dur_avg = iter_opt_dur_avg + iter_data_dur_avg
            
            remaining_opt_time = (FLAGS.iter-it)*iter_opt_dur_avg
            remaining_data_time = (FLAGS.iter-it)*iter_data_dur_avg
            remaining_time = remaining_opt_time + remaining_data_time

            print("iter=%5d, main_loss=%.6f, reg_loss=%.6f, motion_loss=%.6f, texture_light_lr=%.4f, geometry_lr=%.4f, motion_lr=%.4f, time=%.1f ms=(%.1f)+(%.1f) ms, rem=%s=(%s)+(%s)" % 
                (it, main_loss_avg, reg_loss_avg, motion_loss_avg,optimizer.param_groups[0]['lr'], (optimizer_mesh.param_groups[0]['lr'] if optimizer_mesh is not None else -1), (optimizer_motion.param_groups[0]['lr'] if optimizer_motion is not None else -1), iter_dur_avg*1000, iter_data_dur_avg*1000, iter_opt_dur_avg*1000, util.time_to_text(remaining_time), util.time_to_text(remaining_data_time), util.time_to_text(remaining_opt_time)))

            wandb.log({
                **{'train/main_loss': main_loss_avg, 'train/reg_loss': reg_loss_avg, 'train/motion_loss': motion_loss_avg},
                **{'train/learning_rate/texture_light_lr': optimizer.param_groups[0]['lr'], 'train/learning_rate/geometry_lr': (optimizer_mesh.param_groups[0]['lr'] if optimizer_mesh is not None else -1), 'train/learning_rate/motion_lr': (optimizer_motion.param_groups[0]['lr'] if optimizer_motion is not None and not no_motion_update else -1)},
                **{f"train/loss_weights/{k}": v for k, v in loss_weights.items()},
                **{f"train/main_losses/{k}": v for k, v in main_losses.items()},
                **{f"train/reg_losses/{k}": v for k, v in reg_losses.items()},
                **{f"train/motion_losses/{k}": v for k,v in motion_losses.items()},
            }, step=it)
            wandb.log(
                {"time-train/ms_per_iter": iter_dur_avg*1000, "time-train/ms_per_data_iter": iter_data_dur_avg*1000, "time-train/ms_per_opt_iter": iter_opt_dur_avg*1000, "time-train/iter_per_sec": 1.0/iter_dur_avg}, step=it
            )

        # ==============================================================================================
        #  run validation on the fly
        # ==============================================================================================
        if it > 0 and it % FLAGS.validate_interval == 0 and FLAGS.local_rank == 0:
            print(f"validate @ iter {it}")
            # [NOTE] do not save gts repeatedly
            if save_out_base_dir_with_gt is None:
                save_out_base_dir_with_gt = os.path.join(FLAGS.out_dir, "validate_otf", f"iter_{it}") 
                save_gts = True
                print(f"save gts to {save_out_base_dir_with_gt}")
            else:
                save_gts = False
                print(f"do not save gts, {save_out_base_dir_with_gt} exists")


            is_better_than_previous = False

            # for each validation dataset, we validate it.
            for dataset_validate_ in dataset_validate:
                item_name = f"validate_{dataset_validate_.split}"
                save_out_dir = os.path.join(FLAGS.out_dir, "validate_otf", f"iter_{it}", item_name)

                metrics_value_dict = validate(glctx, geometry, opt_material, lgt, dataset_validate_, save_out_dir, FLAGS, save_gts)

                if not save_gts:
                    GT_NAMES = ['ref', 'ref_alpha', 'mask_at_box']
                    for gt_dir_name in GT_NAMES:
                        src_gt_dir = os.path.join(save_out_base_dir_with_gt, item_name, gt_dir_name)
                        dst_gt_dir = os.path.join(save_out_dir, gt_dir_name)
                        if os.path.exists(src_gt_dir):
                            _src_gt_dir = os.path.relpath(src_gt_dir, os.path.dirname(dst_gt_dir))
                            if os.path.exists(dst_gt_dir):
                                os.remove(dst_gt_dir)
                            os.symlink(_src_gt_dir, dst_gt_dir)
                            print(f"symlink {src_gt_dir} to {dst_gt_dir}")
                        else:
                            print(f"src_gt_dir {src_gt_dir} does not exist???")
                metric_log_dict = {}
                for k, v in metrics_value_dict.items():
                    metric_name = f"metrics_otf/{k}/{item_name}"
                    metric_log_dict.update({metric_name: v})

                    # Monitor certain metricss
                    if metric_name == monitor_metric_name:
                        # [XXX] If the monitor_metric_name is not the first one, 
                        # then the metric type before it will not be recored as 
                        # metrics_otf_better. By default, we use the first one.
                        # i.e., "metrics_otf/psnr/validate_${validate_dataset.0.split}"
                        # [TODO]
                        is_better_than_previous = metric_compare_func(v, better_metric_previous_value)
                        if is_better_than_previous:
                            # save the better ckpt
                            print(f"is_better_than_previous! monitor_metric {monitor_metric_name}, {better_metric_previous_value} -> {v}, saving it!")
                            training.save_ckpt(outlier_ckpt_dir, it, trainer_noddp, optimizer, optimizer_mesh, optimizer_motion, best=True)
                            better_metric_previous_value = v

                        if metric_compare_func(outlier_metric_previous_value , monitor_metric_outlier_threshold + v):
                            print(f"outlier_metric {metric_name} is {v}, previous value is {outlier_metric_previous_value}, threshold is {monitor_metric_outlier_threshold}, saving outlier")

                            training.save_ckpt(outlier_ckpt_dir, it, trainer_noddp, optimizer, optimizer_mesh, optimizer_motion)
                            # training.clean_up_ckpt(ckpt_dir, FLAGS.num_kept_ckpts)

                            bare_mesh = geometry.getMesh(None, None)
                            if hasattr(geometry, "lbs_weights"):
                                bare_mesh.v_color = util.skin_weights2color(geometry.lbs_weights)
                            outlier_mesh_dir_it = os.path.join(outlier_mesh_dir, f"iter_{it}")
                            os.makedirs(outlier_mesh_dir_it, exist_ok=True)
                            obj.write_obj(outlier_mesh_dir_it, bare_mesh, save_material=False, save_uv=False, save_v_color=True)
                            print(f"save outlier mesh to {outlier_mesh_dir_it}")
                        else:
                            print("outlier_metric {metric_name} is not outlier, skip saving outlier.")
                            outlier_metric_previous_value = v
                wandb.log(metric_log_dict, step=it)

                # after check is_better_than_previous, log all better metrics
                if is_better_than_previous:
                    # log better metrics
                    metric_log_dict = {}
                    for k, v in metrics_value_dict.items():
                        better_metric_name = f"metrics_otf_better/{k}/{item_name}"
                        metric_log_dict.update({better_metric_name: v})
                    wandb.log(metric_log_dict, step=it)

            if is_better_than_previous: 
                # after check is_better_than_previous, save better results
                # soft link the better save_dir
                src_save_out_dir = os.path.join(FLAGS.out_dir, "validate_otf", f"iter_{it}")
                dst_save_out_dir = os.path.join(FLAGS.out_dir, "validate_otf", "better")
                if os.path.exists(dst_save_out_dir):
                    os.remove(dst_save_out_dir)
                _src_save_out_dir = os.path.relpath(src_save_out_dir, os.path.dirname(dst_save_out_dir))
                os.symlink(_src_save_out_dir, dst_save_out_dir)
                print(f"is_better_than_previous! symlink {src_save_out_dir} to {dst_save_out_dir}")

        # ==============================================================================================
        #  save checkpoints
        # ==============================================================================================
        if it > 0 and it % FLAGS.save_ckpt_interval == 0 and FLAGS.local_rank == 0:
            training.save_ckpt(ckpt_dir, it, trainer_noddp, optimizer, optimizer_mesh, optimizer_motion)
            training.clean_up_ckpt(ckpt_dir, FLAGS.num_kept_ckpts)
        
        # ==============================================================================================
        # save mesh
        # ==============================================================================================
        if it > 0 and it % FLAGS.save_mesh_interval == 0 and FLAGS.local_rank == 0:
            bare_mesh = geometry.getMesh(None, None)

            # write blend skinning mesh
            if hasattr(geometry, "lbs_weights"):
                bare_mesh.v_color = util.skin_weights2color(geometry.lbs_weights)

            mesh_dir_it = os.path.join(mesh_dir, f"iter_{it}")
            os.makedirs(mesh_dir_it, exist_ok=True)
            obj.write_obj(mesh_dir_it, bare_mesh, save_material=False, save_uv=False, save_v_color=True)
            print(f"save mesh to {mesh_dir_it}")

        if FLAGS.multi_gpu:
            torch.distributed.barrier()

        iter_data_start_time = time.time()

    return geometry, opt_material

def build_scheds(FLAGS, lr_schedule, optimizer_mesh, optimizer_motion, optimizer, last_step):
    if optimizer is not None:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_schedule(x), last_epoch=last_step) 
        print(f"Using Adam optimizer for material and light")
    else:
        scheduler = None
    if optimizer_mesh is not None:
        scheduler_mesh = torch.optim.lr_scheduler.LambdaLR(optimizer_mesh, lr_lambda=lambda x: lr_schedule(x), last_epoch=last_step) 
        print(f"Using {FLAGS.optimizer_geometry} optimizer for geometry")
    else:
        scheduler_mesh = None
    if optimizer_motion is not None:
        scheduler_motion = torch.optim.lr_scheduler.LambdaLR(optimizer_motion, lr_lambda=lambda x: lr_schedule(x), last_epoch=last_step)
        print("Using Adam optimizer for motion")
    else:
        scheduler_motion = None
    return scheduler, scheduler_mesh, scheduler_motion

def build_optims(trainer_noddp, FLAGS, optimize_geometry, learning_rate_pos=None, learning_rate_mat=None, learning_rate_skn=None):
    if FLAGS.multi_gpu: 
        # Multi GPU training mode
        import apex
        from apex.parallel import DistributedDataParallel as DDP

        trainer = DDP(trainer_noddp)
        trainer.train()
        if optimize_geometry and learning_rate_pos is not None:
            if isinstance(trainer_noddp.geometry, DMTetGeometryDyn):
                FLAGS.optimizer_geometry = 'adam'
                print(f"DMTetGeometryDyn not supports vectoradam, using adam instead")

            if FLAGS.optimizer_geometry == 'adam':
                optimizer_mesh = apex.optimizers.FusedAdam(trainer_noddp.geo_params, lr=learning_rate_pos)
            elif FLAGS.optimizer_geometry == 'vectoradam':
                optimizer_mesh = VectorAdam(trainer_noddp.geo_params, lr=learning_rate_pos)
        else:
            optimizer_mesh = None

        if len(trainer_noddp.motion_params) > 0 and learning_rate_skn is not None:
            optimizer_motion = apex.optimizers.FusedAdam(trainer_noddp.motion_params, lr=learning_rate_skn)
        else:
            optimizer_motion = None

        if learning_rate_mat is not None:
            optimizer = apex.optimizers.FusedAdam(trainer_noddp.params, lr=learning_rate_mat)
        else:
            optimizer = None

        set_seed(FLAGS.seed + int(FLAGS.rank), FLAGS.strict_reproducibility)
        print(f"[rank {FLAGS.rank}] set seed to {FLAGS.seed + int(FLAGS.rank)}")
    else:
        # Single GPU training mode
        trainer = trainer_noddp
        if optimize_geometry and learning_rate_pos is not None:
            if isinstance(trainer_noddp.geometry, DMTetGeometryDyn):
                FLAGS.optimizer_geometry = 'adam'
                print(f"DMTetGeometryDyn not supports vectoradam, using adam instead")

            if FLAGS.optimizer_geometry == 'adam':
                optimizer_mesh = torch.optim.Adam(trainer_noddp.geo_params, lr=learning_rate_pos)
            elif FLAGS.optimizer_geometry == 'vectoradam':
                optimizer_mesh = VectorAdam(trainer_noddp.geo_params, lr=learning_rate_pos)
        else:
            optimizer_mesh = None

        if len(trainer_noddp.motion_params) > 0 and learning_rate_skn is not None:
            optimizer_motion = torch.optim.Adam(trainer_noddp.motion_params, lr=learning_rate_skn)
        else:
            optimizer_motion = None

        if learning_rate_mat is not None:
            optimizer = torch.optim.Adam(trainer_noddp.params, lr=learning_rate_mat)
        else:
            optimizer = None
    return trainer, optimizer_mesh, optimizer_motion, optimizer

#----------------------------------------------------------------------------
# Main function.
#----------------------------------------------------------------------------

def main(FLAGS):
    _check_cfg(FLAGS)
    set_seed(FLAGS.seed, FLAGS.strict_reproducibility)
    print(f"set seed to {FLAGS.seed}")

    FLAGS.multi_gpu = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    if FLAGS.multi_gpu:
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = 'localhost'
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = str(_find_free_port())
        print(f"MASTER_ADDR: {os.environ['MASTER_ADDR']}")
        print(f"MASTER_PORT: {os.environ['MASTER_PORT']}")

        FLAGS.world_size = int(os.environ["WORLD_SIZE"])
        FLAGS.rank = int(os.environ["RANK"])
        FLAGS.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(FLAGS.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

        print(f"actual interations: from {FLAGS.iter} to {FLAGS.iter * FLAGS.world_size}")
        # FLAGS.iter = FLAGS.iter * FLAGS.world_size
    else:
        torch.cuda.set_device(FLAGS.local_rank)

    if FLAGS.display_res is None:
        FLAGS.display_res = FLAGS.train_res

    if FLAGS.local_rank == 0:

        os.makedirs(FLAGS.out_dir, exist_ok=True)
        print(f"output directory: {FLAGS.out_dir}")

        wandb.init(project=FLAGS.exp_name, name=FLAGS.out_dir, 
            config=OmegaConf.to_container(FLAGS, resolve=True, throw_on_missing=True), dir=FLAGS.out_dir
        )

    # ==============================================================================================
    #  Create data pipeline
    # ==============================================================================================
    if FLAGS.get("use_legacy_dataset", False):
        LOGGER.warning("Use legacy dataset")

        from dataset.legacy.dataset_mesh import DatasetMesh
        from dataset.legacy.dataset_nerf import DatasetNERF
        from dataset.legacy.dataset_llff import DatasetLLFF
        from dataset.legacy.dataset_zju_mocap import DatasetZJUMocap

        if os.path.splitext(FLAGS.ref_mesh)[1] == '.obj':
            ref_mesh         = mesh.load_mesh(FLAGS.ref_mesh, FLAGS.mtl_override)
            dataset_train    = DatasetMesh(ref_mesh, glctx, RADIUS, FLAGS, validate=False)
            dataset_validate = DatasetMesh(ref_mesh, glctx, RADIUS, FLAGS, validate=True)
        elif os.path.isdir(FLAGS.ref_mesh):
            if os.path.isfile(os.path.join(FLAGS.ref_mesh, 'poses_bounds.npy')):
                dataset_train    = DatasetLLFF(FLAGS.ref_mesh, FLAGS, examples=(FLAGS.iter+1)*FLAGS.batch)
                dataset_validate = DatasetLLFF(FLAGS.ref_mesh, FLAGS)
            elif os.path.isfile(os.path.join(FLAGS.ref_mesh, 'transforms_train.json')):
                dataset_train    = DatasetNERF(os.path.join(FLAGS.ref_mesh, 'transforms_train.json'), FLAGS, examples=(FLAGS.iter+1)*FLAGS.batch)
                dataset_validate = DatasetNERF(os.path.join(FLAGS.ref_mesh, 'transforms_test.json'), FLAGS)
            elif os.path.isfile(os.path.join(FLAGS.ref_mesh, 'annots.npy')):
                if not FLAGS.no_train:
                    dataset_train = DatasetZJUMocap(FLAGS, split="train" if not FLAGS.dev_run else "dev_run", examples=(FLAGS.iter+1)*FLAGS.batch)
                else:
                    dataset_train = None

                if FLAGS.dev_run and FLAGS.dev_run_same_train_val:
                    print(f"In dev_run mode.")
                    dataset_validate = [DatasetZJUMocap(FLAGS, split="train" if not FLAGS.dev_run else "dev_run", examples=FLAGS.num_val_examples)]
                else:
                    print(f"validate_splits: {FLAGS.validate_splits}")
                    dataset_validate = [DatasetZJUMocap(FLAGS, split=split, examples=FLAGS.num_val_examples) for split in FLAGS.validate_splits + (["dev_run"] if FLAGS.dev_run else [])]
    else:
        if not FLAGS.no_train:
            dataset_train = hydra.utils.instantiate(FLAGS.train_dataset)
        else:
            dataset_train = None
        
        if isinstance(FLAGS.validate_dataset, DictConfig):
            FLAGS.validate_dataset = ListConfig([FLAGS.validate_dataset])
        dataset_validate = [hydra.utils.instantiate(cfg) for cfg in FLAGS.validate_dataset]

        # [TODO] dev_run

    # ==============================================================================================
    #  Create GPU rasterization context
    # ==============================================================================================
    if FLAGS.rasterize_context == 'opengl':
        print("Using OpenGL rasterizer")
        glctx = dr.RasterizeGLContext()
    elif FLAGS.rasterize_context == 'cuda':
        print("Using CUDA rasterizer")
        glctx = dr.RasterizeCudaContext()
    else:
        raise ValueError(f"Unknown rasterize_context: {FLAGS.rasterize_context}")

    # ==============================================================================================
    #  Create env light with trainable parameters
    # ==============================================================================================

    if FLAGS.learn_light:
        print("Learn light. Use random envmap.")
        lgt = light.create_trainable_env_rnd(512, scale=0.0, bias=0.5)
    else:
        print(f"Not learn light. External envmap: {FLAGS.envmap} and scale: {FLAGS.env_scale}")
        lgt = light.load_env(FLAGS.envmap, scale=FLAGS.env_scale)

    if not FLAGS.use_mesh:
        # ==============================================================================================
        #  If no initial guess, use DMtets to create geometry
        # ==============================================================================================

        # Setup geometry for optimization
        if isinstance(dataset_validate, list):
            animation_meta_data = dataset_validate[0].get_animation_meta_data()
        else:
            animation_meta_data = dataset_validate.get_animation_meta_data()
        geometry = DMTetGeometryDyn(grid_res=FLAGS.dmtet_grid, animation_meta_data=animation_meta_data, learn_skinning=True, learn_non_rigid_offset=FLAGS.learn_non_rigid_offset, FLAGS=FLAGS)

        # Setup textures, make initial guess from reference if possible
        mat = initial_guess_material(geometry, True, FLAGS)
    
        # Run optimization
        if FLAGS.iter > 0:
            geometry, mat = optimize_mesh(glctx, geometry, mat, lgt, dataset_train, dataset_validate, 
                            FLAGS, pass_idx=0, pass_name="dmtet_pass1", warmup_iter=FLAGS.warmup_iter_tet, optimize_light=FLAGS.learn_light, log_interval=FLAGS.log_interval)
        else:
            print("Skipping optimization, no iterations specified.")

        if FLAGS.local_rank == 0 and FLAGS.validate:
            metrics_log_dict = {}
            if isinstance(dataset_validate, list):
                for dataset_validate_ in dataset_validate:
                    item_name = f"dmtet_validate_{dataset_validate_.split}"
                    metrics_value_dict = validate(glctx, geometry, mat, lgt, dataset_validate_, os.path.join(FLAGS.out_dir, item_name), FLAGS)
                    for k, v in metrics_value_dict.items():
                        metrics_log_dict.update({f"metrics/{k}/{item_name}": v})
            else:
                item_name = "dmtet_validate"
                metrics_value_dict = validate(glctx, geometry, mat, lgt, dataset_validate, os.path.join(FLAGS.out_dir, item_name), FLAGS)
                for k, v in metrics_value_dict.items():
                    metrics_log_dict.update({f"metrics/{k}/{item_name}": v})
            wandb.log(metrics_log_dict)

        if FLAGS.no_mesh_export is True:
            return

        # Create textured mesh from result
        if 'kd_ks_normal' in mat and mat['kd_ks_normal'].use_texture_conditional_inputs is True:
            # unset conditional inputs, if condition inputs have batch size 
            # that is different from the batch size of the geometry,
            # the process crushes.
            mat['kd_ks_normal'].register_conditonal_inputs(None)
        base_mesh = xatlas_uvmap(glctx, geometry, mat, FLAGS)

        # Free temporaries / cached memory 
        torch.cuda.empty_cache()
        mat['kd_ks_normal'].cleanup()
        del mat['kd_ks_normal']

        lgt = lgt.clone()
        if isinstance(dataset_validate, list):
            animation_meta_data = dataset_validate[0].get_animation_meta_data()
        else:
            animation_meta_data = dataset_validate.get_animation_meta_data()
        if FLAGS.learn_mesh_skinning is False:
            with torch.no_grad():
                animation_meta_data["lbs_weights"], _ = geometry.query_weights(geometry.getMesh(mat).v_pos)  # Export LBS weights for later use
                animation_meta_data["lbs_weights"] = animation_meta_data["lbs_weights"].to(geometry.verts.dtype)
            geometry = DLMeshDyn(initial_guess=base_mesh, animation_meta_data=animation_meta_data, learn_skinning=False, learn_non_rigid_offset=False, FLAGS=FLAGS)
            geometry.weights = animation_meta_data["lbs_weights"]

        elif FLAGS.learn_mesh_skinning is True:
            dmtet_skin_net = geometry.skin_net
            weights = geometry.weights
            geometry = DLMeshDyn(initial_guess=base_mesh, animation_meta_data=animation_meta_data, learn_skinning=True, learn_non_rigid_offset=False, FLAGS=FLAGS)
            geometry.skin_net = dmtet_skin_net
            geometry.weights = weights
        else:
            raise ValueError(f"Invalid learn_mesh_skinning value, must be True or False, but got %s" % FLAGS.learn_mesh_skinning)

        if FLAGS.local_rank == 0:
            # Dump mesh for debugging.
            os.makedirs(os.path.join(FLAGS.out_dir, "dmtet_mesh"), exist_ok=True)
            obj.write_obj(os.path.join(FLAGS.out_dir, "dmtet_mesh/"), base_mesh)
    
            # write blend skinning mesh
            if hasattr(geometry, "weights"):
                save_dir_path = os.path.join(FLAGS.out_dir, "dmtet_bare_mesh")
                base_mesh.v_color = util.skin_weights2color(geometry.weights)
                os.makedirs(save_dir_path, exist_ok=True)
                obj.write_obj(save_dir_path, base_mesh, save_material=False, save_v_color=True)
    
            light.save_env_map(os.path.join(FLAGS.out_dir, "dmtet_mesh/probe.hdr"), lgt)

        # ==============================================================================================
        #  Pass 2: Train with fixed topology (mesh)
        # ==============================================================================================
        if FLAGS.finetune_tet2mesh is True:
            geometry, mat = optimize_mesh(glctx, geometry, base_mesh.material, lgt, dataset_train, dataset_validate, FLAGS, 
                        pass_idx=1, pass_name="mesh_pass", warmup_iter=FLAGS.warmup_iter_mesh, optimize_light=FLAGS.learn_light and not FLAGS.lock_light, 
                        optimize_geometry=not FLAGS.lock_pos, log_interval=FLAGS.log_interval)
        else:
            # [XXX] skip the final evalution and mesh save
            return
    else:
        # ==============================================================================================
        #  Train with fixed topology (mesh)
        # ==============================================================================================

        # Load initial guess mesh from file
        if isinstance(dataset_validate, list):
            animation_meta_data = dataset_validate[0].get_animation_meta_data()
        else:
            animation_meta_data = dataset_validate.get_animation_meta_data()
        geometry = DLMeshDyn(initial_guess=animation_meta_data, animation_meta_data=animation_meta_data, learn_skinning=FLAGS.learn_mesh_skinning, learn_non_rigid_offset=FLAGS.learn_non_rigid_offset, FLAGS=FLAGS)
        
        mat = initial_guess_material(geometry, FLAGS.learn_mesh_material_with_mlp, FLAGS, init_mat=geometry.initial_guess.material)

        if FLAGS.iter > 0:
            geometry, mat = optimize_mesh(glctx, geometry, mat, lgt, dataset_train, dataset_validate, FLAGS, pass_idx=0, pass_name="mesh_pass", 
                                            warmup_iter=FLAGS.warmup_iter_mesh, optimize_light=not FLAGS.lock_light, optimize_geometry=not FLAGS.lock_pos, log_interval=FLAGS.log_interval)
        else:
            print("Skipping optimization, no iterations specified.")

        # [FIXME] need to finetune texture if use xaltlas
        # if FLAGS.learn_mesh_material_with_mlp:
        #     base_mesh = mesh_uvmap(glctx, geometry, mat, FLAGS)

        #     # Free temporaries / cached memory 
        #     torch.cuda.empty_cache()
        #     mat['kd_ks_normal'].cleanup()
        #     del mat['kd_ks_normal'] 

        #     mat = geometry.mesh.material = base_mesh.material

    # ==============================================================================================
    #  Validate
    # ==============================================================================================
    if FLAGS.validate and FLAGS.local_rank == 0:
        metric_log_dict = {}
        if isinstance(dataset_validate, list):
            for dataset_validate_ in dataset_validate:
                item_name = f"validate_{dataset_validate_.split}"
                metrics_value_dict = validate(glctx, geometry, mat, lgt, dataset_validate_, os.path.join(FLAGS.out_dir, item_name), FLAGS)
                for k, v in metrics_value_dict.items():
                    metric_log_dict.update({f"metrics/{k}/{item_name}": v})
        else:
            item_name = "validate"
            metrics_value_dict = validate(glctx, geometry, mat, lgt, dataset_validate, os.path.join(FLAGS.out_dir, item_name), FLAGS)
            for k, v in metrics_value_dict.items():
                metric_log_dict.update({f"metrics/{k}/{item_name}": v})
        wandb.log(metrics_log_dict)

    # ==============================================================================================
    #  Dump output
    # ==============================================================================================
    if FLAGS.local_rank == 0:
        geometry: DLMeshDyn
        final_mesh = geometry.getMesh(mat)
        os.makedirs(os.path.join(FLAGS.out_dir, "mesh"), exist_ok=True)
        obj.write_obj(os.path.join(FLAGS.out_dir, "mesh/"), final_mesh)
        light.save_env_map(os.path.join(FLAGS.out_dir, "mesh/probe.hdr"), lgt)

###############################################################################
# helper functions
###############################################################################

CONF_FP: str = os.path.join(os.path.dirname(__file__), "configs")

def set_seed(seed, strict_reproducibility=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if strict_reproducibility:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def _find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port

def _check_cfg(FLAGS):
    fail_to_check = False
    if len(FLAGS.subdivide_tetmesh_iters) > 0 and (FLAGS.learn_sdf_with_mlp is True and FLAGS.learn_tet_vert_deform_with_mlp is True):
        # [NOTE] The deformation is scale specific... which means it is irrelavant with SDF, 
        # it is used to cater for the more fine-grained SDF-to-mesh conversion
        raise ValueError("The deformation is scale specific... which means it is irrelavant with SDF, \
                            it is used to cater for the more fine-grained SDF-to-mesh conversion. \
                            subdivision breaks the prediction scale of the MLP.")

@hydra.main(config_path=CONF_FP, config_name="base", version_base="1.2")
def cli(cfg):
    return main(cfg)

#----------------------------------------------------------------------------
# cmd
#----------------------------------------------------------------------------

if __name__ == "__main__":
    cli()

#----------------------------------------------------------------------------
