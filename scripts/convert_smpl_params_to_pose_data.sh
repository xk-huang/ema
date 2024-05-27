#!/bin/sh
# ZJU-Mocap
for i in 313 315; do
python scripts/convert_smpl_params_to_pose_data.py --smpl_dir data/zju_mocap/CoreView_${i}/new_params --base_dir data/zju_mocap/CoreView_${i}/ --smpl_model_path ./data/SMPL_NEUTRAL.pklc --start_idx 1
done

for i in 377 386 387 390 392 393 394; do
python scripts/convert_smpl_params_to_pose_data.py --smpl_dir data/zju_mocap/CoreView_${i}/new_params --base_dir data/zju_mocap/CoreView_${i}/ --smpl_model_path ./data/SMPL_NEUTRAL.pkl
done

# H36M
for i in 1 5 6 7 8 9 11; do 
echo $i
python scripts/convert_smpl_params_to_pose_data.py --smpl_dir data/h36m/S$i/Posing/new_params --base_dir data/h36m/S$i/Posing/ --smpl_model_path ./data/SMPL_NEUTRAL.pkl
done

i=9
python scripts/convert_smpl_params_to_pose_data.py --smpl_dir data/h36m/S$i/Posing/new_params --base_dir data/h36m/S$i/Posing/ --smpl_model_path ./data/SMPL_NEUTRAL.pkl
python scripts/convert_smpl_params_to_pose_data.py --smpl_dir data/h36m/S$i/Posing/new_params --base_dir data/h36m/S$i/Posing_easymocap/ --smpl_model_path ./data/SMPL_NEUTRAL.pkl