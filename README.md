<div align="center">
<h1>
  <b>EMA</b>: <b>E</b>fficient <b>M</b>eshy Neural Fields <br> for Animatable Human <b>A</b>vatars</br>
</h1>

<div>
    <a href="https://xk-huang.github.io/">
        Xiaoke Huang
    </a>&emsp;
    <!-- </br>Tsinghua University -->
    <a href="https://www.linkedin.com/in/yiji-cheng-a8b922213/">
        Yiji Cheng
    </a>&emsp;
    <!-- </br>Tsinghua University -->
    <a href="https://andytang15.github.io/">
        Yansong Tang*
    </a>&emsp;
    <!-- </br>Tsinghua University -->
    <a href="https://scholar.google.com/citations?hl=zh-CN&user=Xrh1OIUAAAAJ&view_op=list_works&sortby=pubdate">
        Xiu Li
    </a>&emsp;
    <!-- </br>Tsinghua University -->
    <a href="https://scholar.google.com/citations?user=6a79aPwAAAAJ&hl=en">
        Jie Zhou
    </a>&emsp;
    <!-- </br>Tsinghua University -->
    <a href="http://ivg.au.tsinghua.edu.cn/Jiwen_Lu/">
        Jiwen Lu
    </a>
    <!-- </br>Tsinghua University -->
    </li>&emsp;
</div>

<strong>Tsinghua University</strong>

<strong><a href='https://xk-huang.github.io/ema/' target='_blank'>Project Page</a></strong>&emsp;
<strong><a href='https://arxiv.org/abs/2303.12965'>Paper (arXiv)</a></strong>&emsp;
<strong><a href='https://xk-huang.github.io/ema/docs/arxiv_Efficient_Meshy_Neural_Fields_for_Animatable_Human_Avatars.small.pdf' target='_blank'>Paper (small)</a></strong>&emsp;
<strong><a href='https://youtu.be/_Wgv7cPJ-Ko' target='_blank'>Video (Unlist)</a></strong>

</div>

![teaser](https://xk-huang.github.io/ema/images/teaser.png)
<!-- <div align="left">
<p>
  <b>EMA</b> efficiently and jointly learns canonical shapes, materials, and motions via differentiable inverse rendering in an end-to-end manner. The method does not require any predefined templates or riggings. The derived avatars are animatable and can be directly applied to the graphics renderer and downstream tasks.
</p>
</div> -->

![pipeline](https://xk-huang.github.io/ema/images/pipeline.png)


## Data

Download the data of ZJU-MoCap from NeuralBody and H36M from Animatable-NeRF (both projects are from ZJU).

We use the old version of the data to train and test our method. However, the new version of data provides better SMPL fitting which we think it could boost the performance of our method.

Store the data in `data/`, which are `data/zju_mocap/` and `data/h36m`.


### Data Preprocessing
Preprocessing data to `"pose_data.pt"`
- Put SMPL neutral in `data/`

```shell
# ZJU-Mocap
# for i in 313 315 377 386 387 390 392 393 394; do
for i in 313 315 377 386; do
python scripts/convert_smpl_params_to_pose_data.py --smpl_dir data/zju_mocap/CoreView_${i}/new_params --base_dir data/zju_mocap/CoreView_${i}/ --smpl_model_path ./data/SMPL_NEUTRAL.pkl
done

# H36M
for i in 1 5 6 9 11; do 
echo $i
python scripts/convert_smpl_params_to_pose_data.py --smpl_dir data/h36m/S$i/Posing/new_params --base_dir data/h36m/S$i/Posing/ --smpl_model_path ./data/SMPL_NEUTRAL.pkl --interval 5
done

i=7
python scripts/convert_smpl_params_to_pose_data.py --smpl_dir data/h36m/S$i/Posing/new_params --base_dir data/h36m/S$i/Posing/ --smpl_model_path ./data/SMPL_NEUTRAL.pkl --interval 5 --num_frames 2916

i=8
python scripts/convert_smpl_params_to_pose_data.py --smpl_dir data/h36m/S$i/Posing/new_params --base_dir data/h36m/S$i/Posing/ --smpl_model_path ./data/SMPL_NEUTRAL.pkl --interval 5 --num_frames 1681
```

### The template tets

Download the [meshes](https://huggingface.co/xk-huang/quartet_meshes) Place them into `tmp/quartet/meshes/`.

## Env

Build docker image.

```shell
cd docker
./make_image.sh ema:v1

```

Start the container:

```shell
docker run -itd \
--name ema-container \
--gpus all \
-v $(realpath .):$(realpath .) \
-v $(realpath ./data/zju_mocap/):$(realpath .)/data/zju_mocap \
-v $(realpath ./data/h36m/):$(realpath .)/data/h36m \
-w $(realpath .)  \
--network=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
ema:v1 bash

docker exec -it ema-container bash
```

## Demo

Download the [checkpoint](https://huggingface.co/xk-huang/ema-base_config-1.zju_mocap.logl2-zju_mocap-313-230306_084623-ckpts/tree/main) into `OUR_DIR`
```shell
OUT_DIR=outputs/demo/
# On 171


WANDB_MODE=offline python visualize_inference.py \
--config-dir ${OUT_DIR}/hydra_config/ \
--config-name config \
external_ckpt_dir=${OUT_DIR}/ \
validate=true \
display=null \
out_dir=outputs/visualize \
no_train=true \
pre_load=false \
background=grey \
num_val_examples=null \
learn_non_rigid_offset=false \
validate_dataset.0.frame_interval=1 \
validate_dataset.0.begin_frame_id=0 \
validate_dataset.0.num_frames=1000 \
validate_dataset.0.order_camera_first=False \
validate_dataset.0.no_image_inputs=True \
validate_dataset.1.frame_interval=1 \
validate_dataset.1.begin_frame_id=0 \
validate_dataset.1.num_frames=1000 \
validate_dataset.1.order_camera_first=False \
validate_dataset.1.no_image_inputs=True \
num_workers_validate=10
```

## Run Codes

### Dev Run

Make sure everything is ok.

```
WANDB_MODE=offline \
python train_dyn.py exp_name=debug tet_dir=tmp/quartet/meshes/ lpips_loss_weight=1.0 iter=1 batch=1
```
We use wandb to minitor the performance. If you do not want to log with wandb, set `WANDB_MODE=offline`.

### Training

Use `meta_configs`, e.g., `meta_configs/base_config-1.h36m.yaml` and `meta_configs/base_config-1.zju_mocap.yaml`:

Example codes:

```shell
CUDA_VISIBLE_DEVICES=0 python scripts/config_sweeper.py -s train_dyn.py -f $META_CONFIG -n \
    +train_dataset.texture_obj_path=data/toy_data_zju_format/SMPL/SMPL_female_default_resolution.obj +train_dataset.force_generate_synthesis_data=false train_dataset._target_=dataset.zju_mocap_synthesis.DatasetZJUMoCapSynthesis +validate_dataset.0.texture_obj_path='data/toy_data_zju_format/SMPL/SMPL_female_default_resolution.obj' validate_dataset.0._target_=dataset.zju_mocap_synthesis.DatasetZJUMoCapSynthesis +validate_dataset.0.force_generate_synthesis_data=false +validate_dataset.1.texture_obj_path='data/toy_data_zju_format/SMPL/SMPL_female_default_resolution.obj' validate_dataset.1._target_=dataset.zju_mocap_synthesis.DatasetZJUMoCapSynthesis +validate_dataset.1.force_generate_synthesis_data=false \
    +train_dataset.noise_scale=0.01 +train_dataset.add_noise=true

python scripts/config_sweeper.py -g='(5,)' -s train_dyn.py -f $META_CONFIG -n \
    +train_dataset.texture_obj_path=data/toy_data_zju_format/SMPL/SMPL_female_default_resolution.obj +train_dataset.force_generate_synthesis_data=false train_dataset._target_=dataset.zju_mocap_synthesis.DatasetZJUMoCapSynthesis +validate_dataset.0.texture_obj_path='data/toy_data_zju_format/SMPL/SMPL_female_default_resolution.obj' validate_dataset.0._target_=dataset.zju_mocap_synthesis.DatasetZJUMoCapSynthesis +validate_dataset.0.force_generate_synthesis_data=false +validate_dataset.1.texture_obj_path='data/toy_data_zju_format/SMPL/SMPL_female_default_resolution.obj' validate_dataset.1._target_=dataset.zju_mocap_synthesis.DatasetZJUMoCapSynthesis +validate_dataset.1.force_generate_synthesis_data=false \
    +train_dataset.noise_scale=0.2 +train_dataset.add_noise=true
```

### Evaluation

may need to use `+no_mesh_export`. backward incompatible

```shell
OUT_DIR=/path/to/results


WANDB_MODE=offline \
CUDA_VISIBLE_DEVICES=0 \
python train_dyn.py --config-dir ${OUT_DIR}/.hydra/ --config-name config \
external_ckpt_dir=${OUT_DIR}/ckpts/ \
display=null \
validate=true \
compute_val_metrics=false \
save_val_images=false \
no_train=true \
no_mesh_export=true \
learn_non_rigid_offset=true \
num_workers_validate=10 \
out_dir='outputs/${exp_name}/eval/${subject_id}/${now:%y%m%d_%H%M%S}' \
+runtime_breakdown=True \
validate_dataset.0.frame_interval=1 \
validate_dataset.0.begin_frame_id=0 \
validate_dataset.0.num_frames=10 \
validate_dataset.0.order_camera_first=False \
validate_dataset.1.frame_interval=1 \
validate_dataset.1.begin_frame_id=0 \
validate_dataset.1.num_frames=10 \
validate_dataset.1.order_camera_first=False \
num_workers_validate=10 \
+validate_update_base_mesh=True \

++learning_rate_schedule_type=linear \
++learning_rate_schedule_steps=[] \
++mask_at_box_bound_pad=0.0 \
++use_vitruvian_pose=true \
subject_id=??? # for efficiency overrite subject_id
+dataset@_global_=???
```

### Metrics

First install `bc`: `apt-get install bc`
Then run metrics: `bash scripts/compute_avarge_metric.sh`

```shell
python scripts/metric_psnr_ssmi_lpips.py \
-t outputs/.../opt \
-g outputs/.../ref \
--exp_name metric \
--log_file tmp/metrics.log
```


### Mesh Rendering Demo

```shell
OUT_DIR=
MESH_PATH=
MTL_PATH=null

WANDB_MODE=offline \
CUDA_VISIBLE_DEVICES=0 \
python demo_mesh_rendering.py --config-dir ${OUT_DIR}/.hydra/ --config-name config \
external_ckpt_dir=${OUT_DIR}/ckpts/ \
display=null \
validate=true \
compute_val_metrics=false \
save_val_images=true \
no_train=true \
no_mesh_export=true \
learn_non_rigid_offset=true \
out_dir='outputs/${exp_name}/mesh/${subject_id}/${now:%y%m%d_%H%M%S}' \
use_mesh=True \
learn_mesh_skinning=True \
learn_mesh_material_with_mlp=True \
++external_mtl_path=$MTL_PATH \
++use_tet_aabb_in_mesh=true \
++external_mesh_path=$MESH_PATH

OUT_DIR=path/to/results
MESH_PATH=tmp/230308.demo_mesh/mesh.uv_unwarp-manually.obj
MTL_PATH=tmp/230308.demo_mesh/sunflower.mtl

```

AIST

```shell
OUT_DIR=

WANDB_MODE=offline \
CUDA_VISIBLE_DEVICES=0 \
python demo_mesh_rendering.py --config-dir ${OUT_DIR}/.hydra/ --config-name config \
external_ckpt_dir=${OUT_DIR}/outlier_ckpts/ \
display=null \
validate=true \
compute_val_metrics=false \
save_val_images=true \
no_train=true \
no_mesh_export=true \
learn_non_rigid_offset=true \
out_dir='outputs/${exp_name}/aist/${subject_id}/${now:%y%m%d_%H%M%S}' \
use_mesh=False \
learn_mesh_skinning=True \
learn_mesh_material_with_mlp=True \
++external_mtl_path=null \
++use_tet_aabb_in_mesh=true \
++use_vitruvian_pose=True \
+dataset@_global_=demo/aist-h36m.yaml


OUT_DIR=path/to/results


# aist gBR_sBM_cAll_d04_mBR1_ch10
# 01 left right cross hand
# 02 left right cross legs
# 03 swimming
# 04 left right
# 05 left right and touch ground
# 06 left right cross hand
# 07 slip on the ground
# 08 slip on the ground
# 09 

+dataset@_global_=demo/aist-h36m.yaml aist_pose_pkl_path=data/aist/motions/
gBR_sBM_cAll_d04_mBR0_ch04
gWA_sBM_cAll_d25_mWA1_ch02
gLH_sFM_cAll_d16_mLH5_ch06
gLO_sBM_cAll_d14_mLO5_ch02
gMH_sBM_cAll_d24_mMH2_ch08
gWA_sFM_cAll_d26_mWA3_ch11
```

## Citation

If you find this repository useful, please consider giving a star ⭐ and citation 🦖:

```
@article{Huang2023EMA,
    title={Efficient Meshy Neural Fields for Animatable Human Avatars},
    author={Xiaoke Huang and Yiji Cheng and Yansong Tang and Xiu Li and Jie Zhou and Jiwen Lu},
    journal={arXiv},
    year={2023},
    volume={abs/2303.12965}
}
```