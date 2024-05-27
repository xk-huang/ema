cnt=0
for OUT_DIR in \
outputs/base_config-all-smpl_init-meta_skin_net-smpl_surface_skinning_reg-full-loss-finetune_mat-full_subdiv.zju_mocap/zju_mocap/315/230301_160500 \
outputs/base_config-all-smpl_init-meta_skin_net-smpl_surface_skinning_reg-full-loss-finetune_mat-full_subdiv.zju_mocap/zju_mocap/377/230301_160511 \
outputs/base_config-all-smpl_init-meta_skin_net-smpl_surface_skinning_reg-full-loss-finetune_mat-full_subdiv.zju_mocap/zju_mocap/386/230301_160522 \
outputs/base_config-all-smpl_init-meta_skin_net-smpl_surface_skinning_reg-full-loss-finetune_mat-full_subdiv.zju_mocap/zju_mocap/387/230303_160721 \
outputs/base_config-all-smpl_init-meta_skin_net-smpl_surface_skinning_reg-full-loss-finetune_mat-full_subdiv.zju_mocap/zju_mocap/390/230303_160732 \
outputs/base_config-all-smpl_init-meta_skin_net-smpl_surface_skinning_reg-full-loss-finetune_mat-full_subdiv.zju_mocap/zju_mocap/392/230303_160743 \
outputs/base_config-all-smpl_init-meta_skin_net-smpl_surface_skinning_reg-full-loss-finetune_mat-full_subdiv.zju_mocap/zju_mocap/393/230303_160754 \
outputs/base_config-all-smpl_init-meta_skin_net-smpl_surface_skinning_reg-full-loss-finetune_mat-full_subdiv.zju_mocap/zju_mocap/394/230303_160805 \

do
# echo \
subject_id=`echo $OUT_DIR | cut -d'/' -f4`
WANDB_MODE=offline \
CUDA_VISIBLE_DEVICES=$cnt \
python train_dyn.py --config-dir ${OUT_DIR}/.hydra/ --config-name config \
external_ckpt_dir=${OUT_DIR}/outlier_ckpts/ \
display=[{"bsdf":"kd"},{"bsdf":"ks"},{"bsdf":"normal"},{"latlong":true}] \
validate=true \
compute_val_metrics=false \
save_val_images=true \
no_train=true \
no_mesh_export=true \
learn_non_rigid_offset=true \
out_dir='outputs/${exp_name}/demo/${subject_id}/${now:%y%m%d_%H%M%S}' \
+dataset@_global_=demo/efficiency-zju_mocap.yaml \
++mask_at_box_bound_pad=0.0 \
++learning_rate_schedule_type=linear \
++use_vitruvian_pose=true \
subject_id=$subject_id \
++learning_rate_schedule_steps=[] &
cnt=$((cnt+1))
done

exit

echo "" > tmp/ffmpeg.sh
for video_type in opt kd ks normal; do
    for i in  `find outputs/demo-efficiency-zju_mocap/demo -type d -name $video_type`; do
    subject_id=`echo $i | cut -d'/' -f4`
    echo ffmpeg -y -framerate 30 -pattern_type glob -i \'"$i/"'*.png'\'  -b:v 4M   -c:v libx264 -pix_fmt yuv420p outputs/demo-efficiency-zju_mocap/demo/$subject_id.$video_type.mp4 >> tmp/ffmpeg.sh
    done
done

# ffmpeg -y -framerate 30 -pattern_type glob -i 'outputs/demo-efficiency-zju_mocap/demo/315/230314_082627/dmtet_validate_novel_view/opt/*.png'  -b:v 4M   -c:v libx264 -pix_fmt yuv420p outputs/demo-efficiency-zju_mocap/demo/315.mp4
# ffmpeg -y -framerate 30 -pattern_type glob -i 'outputs/demo-efficiency-zju_mocap/demo/392/230314_082627/dmtet_validate_novel_view/opt/*.png'  -b:v 4M   -c:v libx264 -pix_fmt yuv420p outputs/demo-efficiency-zju_mocap/demo/392.mp4
# ffmpeg -y -framerate 30 -pattern_type glob -i 'outputs/demo-efficiency-zju_mocap/demo/313/230314_074545/dmtet_validate_novel_view/opt/*.png'  -b:v 4M   -c:v libx264 -pix_fmt yuv420p outputs/demo-efficiency-zju_mocap/demo/313.mp4
# ffmpeg -y -framerate 30 -pattern_type glob -i 'outputs/demo-efficiency-zju_mocap/demo/394/230314_082627/dmtet_validate_novel_view/opt/*.png'  -b:v 4M   -c:v libx264 -pix_fmt yuv420p outputs/demo-efficiency-zju_mocap/demo/394.mp4
# ffmpeg -y -framerate 30 -pattern_type glob -i 'outputs/demo-efficiency-zju_mocap/demo/387/230314_082626/dmtet_validate_novel_view/opt/*.png'  -b:v 4M   -c:v libx264 -pix_fmt yuv420p outputs/demo-efficiency-zju_mocap/demo/387.mp4
# ffmpeg -y -framerate 30 -pattern_type glob -i 'outputs/demo-efficiency-zju_mocap/demo/386/230314_082627/dmtet_validate_novel_view/opt/*.png'  -b:v 4M   -c:v libx264 -pix_fmt yuv420p outputs/demo-efficiency-zju_mocap/demo/386.mp4
# ffmpeg -y -framerate 30 -pattern_type glob -i 'outputs/demo-efficiency-zju_mocap/demo/390/230314_082626/dmtet_validate_novel_view/opt/*.png'  -b:v 4M   -c:v libx264 -pix_fmt yuv420p outputs/demo-efficiency-zju_mocap/demo/390.mp4
# ffmpeg -y -framerate 30 -pattern_type glob -i 'outputs/demo-efficiency-zju_mocap/demo/393/230314_082626/dmtet_validate_novel_view/opt/*.png'  -b:v 4M   -c:v libx264 -pix_fmt yuv420p outputs/demo-efficiency-zju_mocap/demo/393.mp4
# ffmpeg -y -framerate 30 -pattern_type glob -i 'outputs/demo-efficiency-zju_mocap/demo/377/230314_082627/dmtet_validate_novel_view/opt/*.png'  -b:v 4M   -c:v libx264 -pix_fmt yuv420p outputs/demo-efficiency-zju_mocap/demo/377.mp4

