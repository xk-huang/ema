OUT_DIR=outputs/base_config-all-smpl_init-meta_skin_net-smpl_surface_skinning_reg-full-loss-finetune_mat-full_subdiv.h36m/h36m/6/230301_194640
MESH_PATH=outputs/base_config-all-smpl_init-meta_skin_net-smpl_surface_skinning_reg-full-loss-finetune_mat-full_subdiv.h36m/h36m/6/230301_194640/dmtet_mesh/mesh.obj

# mesh
WANDB_MODE=offline \
CUDA_VISIBLE_DEVICES=0 \
python -m debugpy --listen localhost:56789 demo_mesh_rendering.py --config-dir ${OUT_DIR}/.hydra/ --config-name config \
external_ckpt_dir=${OUT_DIR}/outlier_ckpts/ \
display=null \
validate=true \
compute_val_metrics=true \
save_val_images=true \
no_train=true \
no_mesh_export=true \
learn_non_rigid_offset=true \
out_dir='outputs/${exp_name}/mesh/${subject_id}/${now:%y%m%d_%H%M%S}' \
use_mesh=True \
learn_mesh_skinning=True \
learn_mesh_material_with_mlp=True \
+external_mtl_path=null \
+use_tet_aabb_in_mesh=true \
+external_mesh_path=$MESH_PATH

WANDB_MODE=offline \
CUDA_VISIBLE_DEVICES=0 \
python -m debugpy --listen localhost:56789 demo_mesh_rendering.py --config-dir ${OUT_DIR}/.hydra/ --config-name config \
external_ckpt_dir=${OUT_DIR}/outlier_ckpts/ \
display=[{"bsdf":"kd"},{"bsdf":"ks"},{"bsdf":"normal"},{"latlong":true}]  \
validate=true \
compute_val_metrics=true \
save_val_images=true \
no_train=true \
no_mesh_export=true \
learn_non_rigid_offset=true \
out_dir='outputs/${exp_name}/mesh/${subject_id}/${now:%y%m%d_%H%M%S}' \
use_mesh=True \
learn_mesh_skinning=True \
learn_mesh_material_with_mlp=True \
+external_mtl_path=null \
+use_tet_aabb_in_mesh=true \
+external_mesh_path=$MESH_PATH \
+dataset@_global_=demo/efficiency-h36m.yaml \
++mask_at_box_bound_pad=0.0 \
++learning_rate_schedule_type=linear \
++use_vitruvian_pose=true \
subject_id=9 \
++learning_rate_schedule_steps=[] 

import os
from render import obj
out_path="tmp/230315.mesh.dlmesh/posed_mesh"
os.makedirs(out_path, exist_ok=True)
obj.write_obj(out_path, opt_mesh, save_material=False, save_uv=False, save_normal=False, save_v_color=False)

out_path="tmp/230315.mesh.dlmesh/canon_mesh"
os.makedirs(out_path, exist_ok=True)
canon_mesh = opt_mesh.clone()
canon_mesh.v_pos = canon_mesh.v_pos_canon
os.makedirs(out_path, exist_ok=True)
obj.write_obj(out_path, canon_mesh, save_material=False, save_uv=False, save_normal=False, save_v_color=False)

out_path="tmp/230315.mesh.dlmesh/"
util.save_image(f"{out_path}/gb_pos.png", (1 + gb_pos.cpu().numpy()[0]) / 2)
util.save_image(f"{out_path}/gb_pos_canon.png", (1 + gb_pos_canon.cpu().numpy()[0]) / 2)

# tet

WANDB_MODE=offline \
CUDA_VISIBLE_DEVICES=0 \
python  -m debugpy --listen localhost:56789  demo_mesh_rendering.py --config-dir ${OUT_DIR}/.hydra/ --config-name config \
external_ckpt_dir=${OUT_DIR}/outlier_ckpts/ \
display=null \
validate=true \
compute_val_metrics=true \
save_val_images=true \
no_train=true \
no_mesh_export=true \
learn_non_rigid_offset=true \
out_dir='outputs/${exp_name}/mesh/${subject_id}/${now:%y%m%d_%H%M%S}' \
use_mesh=False \
learn_mesh_skinning=True \
learn_mesh_material_with_mlp=True \
+external_mtl_path=null \
+use_tet_aabb_in_mesh=true \
+external_mesh_path=$MESH_PATH

import os
from render import obj
out_path="tmp/230315.mesh.dmtet/posed_mesh"
os.makedirs(out_path, exist_ok=True)
obj.write_obj(out_path, opt_mesh, save_material=False, save_uv=False, save_normal=False, save_v_color=False)

out_path="tmp/230315.mesh.dmtet/canon_mesh"
os.makedirs(out_path, exist_ok=True)
canon_mesh = opt_mesh.clone()
canon_mesh.v_pos = canon_mesh.v_pos_canon
os.makedirs(out_path, exist_ok=True)
obj.write_obj(out_path, canon_mesh, save_material=False, save_uv=False, save_normal=False, save_v_color=False)

out_path="tmp/230315.mesh.dmtet/"
util.save_image(f"{out_path}/gb_pos.png", (1 + gb_pos.cpu().numpy()[0]) / 2)
util.save_image(f"{out_path}/gb_pos_canon.png", (1 + gb_pos_canon.cpu().numpy()[0]) / 2)


WANDB_MODE=offline \
CUDA_VISIBLE_DEVICES=0 \
python  -m debugpy --listen localhost:56789  demo_mesh_rendering.py --config-dir ${OUT_DIR}/.hydra/ --config-name config \
external_ckpt_dir=${OUT_DIR}/outlier_ckpts/ \
display=null \
validate=true \
compute_val_metrics=true \
save_val_images=true \
no_train=true \
no_mesh_export=true \
learn_non_rigid_offset=true \
out_dir='outputs/${exp_name}/mesh/${subject_id}/${now:%y%m%d_%H%M%S}' \
use_mesh=False \
learn_mesh_skinning=True \
learn_mesh_material_with_mlp=True \
+external_mtl_path=null \
+use_tet_aabb_in_mesh=true \
+external_mesh_path=$MESH_PATH \
+dataset@_global_=demo/efficiency-h36m.yaml \
++mask_at_box_bound_pad=0.0 \
++learning_rate_schedule_type=linear \
++use_vitruvian_pose=true \
subject_id=9 \
display=[{"bsdf":"kd"},{"bsdf":"ks"},{"bsdf":"normal"},{"latlong":true}]  \
++learning_rate_schedule_steps=[] 