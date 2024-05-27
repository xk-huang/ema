OUT_DIR=\
outputs/base_config-all-smpl_init-all-meta_skin_net-smpl_surface_skinning_reg-full-loss.zju_mocap/zju_mocap/393/230303_160754

WANDB_MODE=offline \
CUDA_VISIBLE_DEVICES=0 \
python train_dyn.py --config-dir ${OUT_DIR}/.hydra/ --config-name config \
external_ckpt_dir=${OUT_DIR}/ckpts/ \
display=null \
validate=true \
compute_val_metrics=false \
save_val_images=true \
no_train=true \
no_mesh_export=true \
learn_non_rigid_offset=true \
++learning_rate_schedule_type='linear' \
++learning_rate_schedule_steps=[] \
out_dir='outputs/${exp_name}/eval/${subject_id}/${now:%y%m%d_%H%M%S}' &

OUT_DIR=\
outputs/base_config-all-smpl_init-all-meta_skin_net-smpl_surface_skinning_reg-full-loss.zju_mocap/zju_mocap/392/230303_160743 

WANDB_MODE=offline \
CUDA_VISIBLE_DEVICES=1 \
python train_dyn.py --config-dir ${OUT_DIR}/.hydra/ --config-name config \
external_ckpt_dir=${OUT_DIR}/ckpts/ \
display=null \
validate=true \
compute_val_metrics=false \
save_val_images=true \
no_train=true \
no_mesh_export=true \
learn_non_rigid_offset=true \
++learning_rate_schedule_type='linear' \
++learning_rate_schedule_steps=[] \
out_dir='outputs/${exp_name}/eval/${subject_id}/${now:%y%m%d_%H%M%S}' &



OUT_DIR=\
outputs/base_config-all-smpl_init-all-meta_skin_net-smpl_surface_skinning_reg-full-loss.zju_mocap/zju_mocap/390/230303_160732 

WANDB_MODE=offline \
CUDA_VISIBLE_DEVICES=2 \
python train_dyn.py --config-dir ${OUT_DIR}/.hydra/ --config-name config \
external_ckpt_dir=${OUT_DIR}/ckpts/ \
display=null \
validate=true \
compute_val_metrics=false \
save_val_images=true \
no_train=true \
no_mesh_export=true \
learn_non_rigid_offset=true \
++learning_rate_schedule_type='linear' \
++learning_rate_schedule_steps=[] \
out_dir='outputs/${exp_name}/eval/${subject_id}/${now:%y%m%d_%H%M%S}' &



OUT_DIR=\
outputs/base_config-all-smpl_init-all-meta_skin_net-smpl_surface_skinning_reg-full-loss.zju_mocap/zju_mocap/387/230303_160721 

WANDB_MODE=offline \
CUDA_VISIBLE_DEVICES=3 \
python train_dyn.py --config-dir ${OUT_DIR}/.hydra/ --config-name config \
external_ckpt_dir=${OUT_DIR}/ckpts/ \
display=null \
validate=true \
compute_val_metrics=false \
save_val_images=true \
no_train=true \
no_mesh_export=true \
learn_non_rigid_offset=true \
++learning_rate_schedule_type='linear' \
++learning_rate_schedule_steps=[] \
out_dir='outputs/${exp_name}/eval/${subject_id}/${now:%y%m%d_%H%M%S}' &



OUT_DIR=\
outputs/base_config-all-smpl_init-all-meta_skin_net-smpl_surface_skinning_reg-full-loss.zju_mocap/zju_mocap/386/230301_031256 

WANDB_MODE=offline \
CUDA_VISIBLE_DEVICES=4 \
python train_dyn.py --config-dir ${OUT_DIR}/.hydra/ --config-name config \
external_ckpt_dir=${OUT_DIR}/ckpts/ \
display=null \
validate=true \
compute_val_metrics=false \
save_val_images=true \
no_train=true \
no_mesh_export=true \
learn_non_rigid_offset=true \
++learning_rate_schedule_type='linear' \
++learning_rate_schedule_steps=[] \
out_dir='outputs/${exp_name}/eval/${subject_id}/${now:%y%m%d_%H%M%S}' &



OUT_DIR=\
outputs/base_config-all-smpl_init-all-meta_skin_net-smpl_surface_skinning_reg-full-loss.zju_mocap/zju_mocap/377/230301_031245 

WANDB_MODE=offline \
CUDA_VISIBLE_DEVICES=5 \
python train_dyn.py --config-dir ${OUT_DIR}/.hydra/ --config-name config \
external_ckpt_dir=${OUT_DIR}/ckpts/ \
display=null \
validate=true \
compute_val_metrics=false \
save_val_images=true \
no_train=true \
no_mesh_export=true \
learn_non_rigid_offset=true \
++learning_rate_schedule_type='linear' \
++learning_rate_schedule_steps=[] \
out_dir='outputs/${exp_name}/eval/${subject_id}/${now:%y%m%d_%H%M%S}' &



OUT_DIR=\
outputs/base_config-all-smpl_init-all-meta_skin_net-smpl_surface_skinning_reg-full-loss.zju_mocap/zju_mocap/315/230301_031233 

WANDB_MODE=offline \
CUDA_VISIBLE_DEVICES=6 \
python train_dyn.py --config-dir ${OUT_DIR}/.hydra/ --config-name config \
external_ckpt_dir=${OUT_DIR}/ckpts/ \
display=null \
validate=true \
compute_val_metrics=false \
save_val_images=true \
no_train=true \
no_mesh_export=true \
learn_non_rigid_offset=true \
++learning_rate_schedule_type='linear' \
++learning_rate_schedule_steps=[] \
out_dir='outputs/${exp_name}/eval/${subject_id}/${now:%y%m%d_%H%M%S}' &



OUT_DIR=\
outputs/base_config-all-smpl_init-all-meta_skin_net-smpl_surface_skinning_reg-full-loss.zju_mocap/zju_mocap/313/230301_031222




WANDB_MODE=offline \
CUDA_VISIBLE_DEVICES=7 \
python train_dyn.py --config-dir ${OUT_DIR}/.hydra/ --config-name config \
external_ckpt_dir=${OUT_DIR}/ckpts/ \
display=null \
validate=true \
compute_val_metrics=false \
save_val_images=true \
no_train=true \
no_mesh_export=true \
learn_non_rigid_offset=true \
++learning_rate_schedule_type='linear' \
++learning_rate_schedule_steps=[] \
out_dir='outputs/${exp_name}/eval/${subject_id}/${now:%y%m%d_%H%M%S}' &

wait


OUT_DIR=\
outputs/base_config-all-smpl_init-all-meta_skin_net-smpl_surface_skinning_reg-full-loss.zju_mocap/zju_mocap/394/230303_160805 

WANDB_MODE=offline \
CUDA_VISIBLE_DEVICES=0 \
python train_dyn.py --config-dir ${OUT_DIR}/.hydra/ --config-name config \
external_ckpt_dir=${OUT_DIR}/ckpts/ \
display=null \
validate=true \
compute_val_metrics=false \
save_val_images=true \
no_train=true \
no_mesh_export=true \
learn_non_rigid_offset=true \
++learning_rate_schedule_type='linear' \
++learning_rate_schedule_steps=[] \
out_dir='outputs/${exp_name}/eval/${subject_id}/${now:%y%m%d_%H%M%S}' 

