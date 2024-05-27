WANDB_MODE=offline \
CUDA_VISIBLE_DEVICES=4 \
python train_dyn.py --config-dir ${OUT_DIR}/.hydra/ --config-name config \
external_ckpt_dir=${OUT_DIR}/outlier_ckpts/ \
display=[{"bsdf":"kd"},{"bsdf":"ks"},{"bsdf":"normal"},{"latlong":true}] \
validate=true \
compute_val_metrics=false \
save_val_images=true \
no_train=true \
no_mesh_export=true \
learn_non_rigid_offset=true \
out_dir='outputs/${exp_name}/demo-313/${subject_id}/${now:%y%m%d_%H%M%S}' \
+dataset@_global_=demo/efficiency-zju_mocap.yaml \
++mask_at_box_bound_pad=0.0 \
++learning_rate_schedule_type=linear \
++use_vitruvian_pose=true \
++learning_rate_schedule_steps=[]


ffmpeg -y -framerate 30 -pattern_type glob -i 'outputs/demo-efficiency-zju_mocap/demo-313/313/230315_024304/dmtet_validate_novel_view/opt/*.png' -b:v 4M -c:v libx264 -pix_fmt yuv420p outputs/demo-efficiency-zju_mocap/demo-313/313.opt.mp4