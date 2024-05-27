OUT_DIR=outputs/base_config-all-smpl_init-meta_skin_net-smpl_surface_skinning_reg-full-loss-finetune_mat-full_subdiv.h36m/h36m/6/230301_194640 \
cnt=0
for PATH_TO_HDR in \
data/light-probes/interior.hdr \
data/light-probes/sunset.hdr \
data/light-probes/studio.hdr \
data/light-probes/sunrise.hdr \
data/light-probes/courtyard.hdr \
data/light-probes/forest.hdr \
data/light-probes/city.hdr \
data/light-probes/night.hdr
do
hdr_name=`basename $PATH_TO_HDR`
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
out_dir='outputs/relight/${subject_id}/'${hdr_name} \
use_mesh=False \
learn_mesh_skinning=True \
learn_mesh_material_with_mlp=True \
++use_tet_aabb_in_mesh=true \
+dataset@_global_=demo/efficiency-h36m.yaml \
++mask_at_box_bound_pad=0.0 \
++use_vitruvian_pose=true \
++external_mtl_path=null \
++external_mesh_path=null \
learn_light=false envmap=$PATH_TO_HDR env_scale=1.0 \
subject_id=9 \
&
cnt=$((cnt+1))
done

wait


OUT_DIR=outputs/base_config-1.ablat-skinning/313/230312_180213 \
cnt=0
for PATH_TO_HDR in \
data/light-probes/interior.hdr \
data/light-probes/sunset.hdr \
data/light-probes/studio.hdr \
data/light-probes/sunrise.hdr \
data/light-probes/courtyard.hdr \
data/light-probes/forest.hdr \
data/light-probes/city.hdr \
data/light-probes/night.hdr
do
hdr_name=`basename $PATH_TO_HDR`
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
out_dir='outputs/relight/${subject_id}/'${hdr_name} \
use_mesh=False \
learn_mesh_skinning=True \
learn_mesh_material_with_mlp=True \
++use_tet_aabb_in_mesh=true \
+dataset@_global_=demo/efficiency-h36m.yaml \
++mask_at_box_bound_pad=0.0 \
++use_vitruvian_pose=true \
++external_mtl_path=null \
++external_mesh_path=null \
learn_light=false envmap=$PATH_TO_HDR env_scale=1.0 \
subject_id=313 \
&
cnt=$((cnt+1))
done

exit

echo "" > tmp/ffmpeg.sh
for i in  `find outputs/relight/ -type d -name 'opt'`; do
subject_id=`echo $i | cut -d'/' -f3`
seq_name=`echo $i | cut -d'/' -f4`
echo ffmpeg -y -framerate 30 -pattern_type glob -i \'"$i/"'*.png'\' -b:v 4M -c:v libx264 -pix_fmt yuv420p outputs/aist/$subject_id.$seq_name.mp4 >> tmp/ffmpeg.sh
done