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

# [FIXME](xk) the script is kind of duplicated against `train_dyn.py`, which only remove the training loop.


RADIUS = 3.0

LOGGER = logging.getLogger(__name__)
print = LOGGER.info

# Enable to debug back-prop anomalies
# torch.autograd.set_detect_anomaly(True)


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
    if eval_mesh.v_tex is None or eval_mesh.t_tex_idx is None:
        print("UV mapping mesh with xatlas...")
        v_pos = eval_mesh.v_pos.detach().cpu().numpy()
        t_pos_idx = eval_mesh.t_pos_idx.detach().cpu().numpy()
        vmapping, indices, uvs = xatlas.parametrize(v_pos, t_pos_idx)

        # Convert to tensors
        indices_int64 = indices.astype(np.uint64, casting='same_kind').view(np.int64)
        
        uvs = torch.tensor(uvs, dtype=torch.float32, device='cuda')
        faces = torch.tensor(indices_int64, dtype=torch.int64, device='cuda')

        new_mesh = mesh.Mesh(v_tex=uvs, t_tex_idx=faces, base=eval_mesh)
    else:
        print("Mesh already has UVs, skipping UV mapping")
        new_mesh = eval_mesh

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

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'metrics.txt'), 'w') as fout:
        fout.write('ID, MSE, PSNR, SSIM, FPS\n')

        print(f"Running validation @ {out_dir}")
        for it, target in enumerate(tqdm(dataloader_validate)):

            # Mix validation background
            target = prepare_batch(target, FLAGS.background)

            result_image, result_dict, extra_dict = validate_itr(glctx, target, geometry, opt_material, lgt, FLAGS, update_base_mesh=False, build_mips=(True if it == 0 else False))
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
    image_loss_fn = None
    lpips_loss_fn = None

    trainer_noddp = Trainer(glctx, geometry, lgt, opt_material, optimize_geometry, optimize_light, image_loss_fn, FLAGS, lpips_loss_fn)

    # Relighting
    if FLAGS.no_train is True and FLAGS.learn_light is False:
            print(f"Do not load lights from checkpoint, using loaded lights {FLAGS.envmap}")
            trainer_noddp.light = None
    
    if FLAGS.external_ckpt_dir is not None:
        # try to load weights if the sdf or deform weights are mis-matched
        try:
            training.resume_from_ckpt(FLAGS.external_ckpt_dir, trainer_noddp, optimizer=None, optimizer_mesh=None, optimizer_motion=None, step=FLAGS.external_ckpt_step, load_geometry=False if FLAGS.use_mesh else True, load_material=False if FLAGS.external_mtl_path is not None else True)
        except RuntimeWarning as e:
            print("Fail to load weights due to shape mismatch. Copy the mismatched weights and retrying...")
            training.resume_from_ckpt(FLAGS.external_ckpt_dir, trainer_noddp, optimizer=None, optimizer_mesh=None, optimizer_motion=None, step=FLAGS.external_ckpt_step, strict=True)
    else:
        LOGGER.warning("external_ckpt_dir is not specified")

    # No training if validate_only is True
    print(f"Rendering only, skipping training.")
    return geometry, opt_material


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
            geometry = DLMeshDyn(initial_guess=base_mesh, animation_meta_data=animation_meta_data, learn_skinning=True, learn_non_rigid_offset=False, FLAGS=FLAGS)
            geometry.skin_net = dmtet_skin_net
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

        if FLAGS.external_mesh_path is not None:
            print(f"Loading external mesh from {FLAGS.external_mesh_path}")
            init_mesh = mesh.load_mesh(FLAGS.external_mesh_path)
        else:
            print("Generating initial guess mesh with meta data in Dataset.")
            init_mesh = mesh.Mesh(v_pos = animation_meta_data["rest_verts_in_canon"], t_pos_idx=animation_meta_data["faces"])

        geometry = DLMeshDyn(initial_guess=init_mesh, animation_meta_data=animation_meta_data, learn_skinning=FLAGS.learn_mesh_skinning, learn_non_rigid_offset=FLAGS.learn_non_rigid_offset, FLAGS=FLAGS)
        
        if FLAGS.external_mtl_path is not None:
            print(f"Loading external mtl from {FLAGS.external_mtl_path}")
            mat = material.load_mtl(FLAGS.external_mtl_path)
            mat = mat[0]
            LOGGER.warning("Only use the first material in the mtl file.")
        else:
            print("generating initial guess material.")
            mat = initial_guess_material(geometry, FLAGS.learn_mesh_material_with_mlp, FLAGS, init_mat=geometry.initial_guess.material)

        # load texture map here.

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
        wandb.log(metric_log_dict)

    # ==============================================================================================
    #  Dump output
    # ==============================================================================================
    if FLAGS.local_rank == 0 and not FLAGS.no_mesh_export:
        geometry: DLMeshDyn
        if getattr(mat, "kd_ks_normal", None) is not None:
            print("has MLP texture, generating UVs with xatlas")
            final_mesh = xatlas_uvmap(glctx, geometry, mat, FLAGS)
        else:
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
