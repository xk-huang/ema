OUT_DIR=outputs/base_config-all-smpl_init-meta_skin_net-smpl_surface_skinning_reg-full-loss-finetune_mat-full_subdiv.h36m/h36m/6/230301_194640
MESH_PATH=tmp/230308.demo_mesh/mesh.uv_unwarp-manually.obj
MTL_PATH=tmp/230308.demo_mesh/sunflower.mtl

WANDB_MODE=offline \
CUDA_VISIBLE_DEVICES=7 \
python demo_mesh_rendering.py --config-dir ${OUT_DIR}/.hydra/ --config-name config \
external_ckpt_dir=${OUT_DIR}/ckpts/ \
display=null \
validate=true \
compute_val_metrics=false \
save_val_images=true \
no_train=true \
no_mesh_export=true \
learn_non_rigid_offset=true \
out_dir='outputs/texture/${subject_id}/${now:%y%m%d_%H%M%S}' \
use_mesh=True \
learn_mesh_skinning=True \
learn_mesh_material_with_mlp=True \
++external_mtl_path=$MTL_PATH \
++use_tet_aabb_in_mesh=true \
+dataset@_global_=demo/efficiency-h36m.yaml \
++mask_at_box_bound_pad=0.0 \
++use_vitruvian_pose=true \
++external_mesh_path=$MESH_PATH



WANDB_MODE=offline \
CUDA_VISIBLE_DEVICES=6 \
python demo_mesh_rendering.py --config-dir ${OUT_DIR}/.hydra/ --config-name config \
external_ckpt_dir=${OUT_DIR}/outlier_ckpts/ \
display=null \
validate=true \
compute_val_metrics=false \
save_val_images=true \
no_train=true \
no_mesh_export=true \
learn_non_rigid_offset=true \
out_dir='outputs/texture-origin/${subject_id}/${now:%y%m%d_%H%M%S}' \
use_mesh=False \
learn_mesh_skinning=True \
learn_mesh_material_with_mlp=True \
++use_tet_aabb_in_mesh=true \
+dataset@_global_=demo/efficiency-h36m.yaml \
++mask_at_box_bound_pad=0.0 \
++use_vitruvian_pose=true \
++external_mtl_path=null \
++external_mesh_path=null

exit

ffmpeg -y -framerate 30 -pattern_type glob -i 'outputs/texture-edited/*.png' -b:v 4M -c:v libx264 -pix_fmt yuv420p outputs/texture-edited.mp4