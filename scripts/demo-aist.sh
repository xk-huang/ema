OUT_DIR=outputs/base_config-all-smpl_init-meta_skin_net-smpl_surface_skinning_reg-full-loss-finetune_mat-full_subdiv.h36m/h36m/6/230301_194640 \
cnt=0
for SEQ_NAME in \
gBR_sBM_cAll_d04_mBR0_ch04 \
gWA_sBM_cAll_d25_mWA1_ch02 \
gLH_sFM_cAll_d16_mLH5_ch06 \
gLO_sBM_cAll_d14_mLO5_ch02 \
gMH_sBM_cAll_d24_mMH2_ch08 \
gWA_sFM_cAll_d26_mWA3_ch11
do

WANDB_MODE=offline \
CUDA_VISIBLE_DEVICES=$cnt \
python demo_mesh_rendering.py --config-dir ${OUT_DIR}/.hydra/ --config-name config \
external_ckpt_dir=${OUT_DIR}/outlier_ckpts/ \
display=null \
validate=true \
compute_val_metrics=false \
save_val_images=true \
no_train=true \
no_mesh_export=true \
learn_non_rigid_offset=true \
out_dir='outputs/aist/${subject_id}/'$SEQ_NAME \
use_mesh=False \
learn_mesh_skinning=True \
learn_mesh_material_with_mlp=True \
++external_mtl_path=null \
++use_tet_aabb_in_mesh=true \
++use_vitruvian_pose=True \
+dataset@_global_=demo/aist-h36m.yaml \
aist_pose_pkl_path=data/aist/motions/$SEQ_NAME.pkl &
cnt=$((cnt+1))
done

wait


OUT_DIR=outputs/base_config-1.ablat-skinning/313/230312_180213 \
cnt=0
for SEQ_NAME in \
gBR_sBM_cAll_d04_mBR0_ch04 \
gWA_sBM_cAll_d25_mWA1_ch02 \
gLH_sFM_cAll_d16_mLH5_ch06 \
gLO_sBM_cAll_d14_mLO5_ch02 \
gMH_sBM_cAll_d24_mMH2_ch08 \
gWA_sFM_cAll_d26_mWA3_ch11
do

WANDB_MODE=offline \
CUDA_VISIBLE_DEVICES=$cnt \
python demo_mesh_rendering.py --config-dir ${OUT_DIR}/.hydra/ --config-name config \
external_ckpt_dir=${OUT_DIR}/outlier_ckpts/ \
display=null \
validate=true \
compute_val_metrics=false \
save_val_images=true \
no_train=true \
no_mesh_export=true \
learn_non_rigid_offset=true \
out_dir='outputs/aist/${subject_id}/'$SEQ_NAME \
use_mesh=False \
learn_mesh_skinning=True \
learn_mesh_material_with_mlp=True \
++external_mtl_path=null \
++use_tet_aabb_in_mesh=true \
++use_vitruvian_pose=True \
+dataset@_global_=demo/aist-zju_mocap.yaml \
aist_pose_pkl_path=data/aist/motions/$SEQ_NAME.pkl &
cnt=$((cnt+1))
done

exit

echo "" > tmp/ffmpeg.sh
for i in  `find outputs/aist/ -type d -name 'opt'`; do
subject_id=`echo $i | cut -d'/' -f3`
seq_name=`echo $i | cut -d'/' -f4`
echo ffmpeg -y -framerate 30 -pattern_type glob -i \'"$i/"'*.png'\' -b:v 4M -c:v libx264 -pix_fmt yuv420p outputs/aist/$subject_id.$seq_name.mp4 >> tmp/ffmpeg.sh
done