# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import os.path as osp
import sys
import argparse
import json

import numpy as np
import torch
import tqdm

sys.path.append(os.path.realpath(os.path.dirname(__file__)))
from utils.body_model import SMPLlayer


VALID_KEYS_IN_JSON= ["poses", "shapes", "Rh", "Th"]

def load_json_to_numpy_dict(fp):
    with open(fp) as f:
        data = json.load(f)
    data = data["annots"][0]
    
    return {
        key: np.array(data[key], dtype=np.float32) for key in VALID_KEYS_IN_JSON
    }

def load_numpy_obj_to_numpy_dict(fp):
    data = np.load(fp, allow_pickle=True).item()
    return {
        key: np.array(data[key], dtype=np.float32) for key in VALID_KEYS_IN_JSON
    }

@torch.no_grad()
def load_rest_pose_info(smpl_dir: int, body_model, start_idx):
    fp = os.path.join(smpl_dir, "{}.{}")
    if osp.exists(_name := fp.format(f"{start_idx}", "json")):
        smpl_data = load_json_to_numpy_dict(_name)
    elif osp.exists(_name := fp.format(f"{start_idx:06d}", "json")):
        smpl_data = load_json_to_numpy_dict(_name)
    elif osp.exists(_name := fp.format(f"{start_idx}", "npy")):
        smpl_data = load_numpy_obj_to_numpy_dict(_name)
    elif osp.exists(_name := fp.format(f"{start_idx:06d}", "npy")):
        smpl_data = load_numpy_obj_to_numpy_dict(_name)
    else:
        raise ValueError(f"cannot find REST pose info: {smpl_dir}, {start_idx}")

    vertices, joints, joints_transform, bones_transform = body_model(
        poses=np.zeros((1, 72), dtype=np.float32),
        shapes=smpl_data["shapes"],
        Rh=np.zeros((1, 3), dtype=np.float32),
        Th=np.zeros((1, 3), dtype=np.float32),
        scale=1,
        new_params=True,
    )
    return (
        vertices.squeeze(0), 
        joints.squeeze(0), 
        joints_transform.squeeze(0),
        bones_transform.squeeze(0),
    )


@torch.no_grad()
def load_pose_info(smpl_dir: int, frame_id: int, body_model):
    fp = os.path.join(smpl_dir, "{}.{}")
    if osp.exists(_name := fp.format(f"{frame_id}", "json")):
        smpl_data = load_json_to_numpy_dict(_name)
    elif osp.exists(_name := fp.format(f"{frame_id:06d}", "json")):
        smpl_data = load_json_to_numpy_dict(_name)
    elif osp.exists(_name := fp.format(f"{frame_id}", "npy")):
        smpl_data = load_numpy_obj_to_numpy_dict(_name)
    elif osp.exists(_name := fp.format(f"{frame_id:06d}", "npy")):
        smpl_data = load_numpy_obj_to_numpy_dict(_name)
    else:
        raise ValueError(f"cannot find pose info: {smpl_dir}, {frame_id}")

    # smpl_data['shapes'] is actually the same across frames (checked)
    vertices, joints, joints_transform, bones_tranform = body_model(
        poses=np.array(smpl_data['poses']),
        shapes=np.array(smpl_data['shapes']),
        Rh=np.array(smpl_data['Rh']),
        Th=np.array(smpl_data['Th']),
        scale=1,
        new_params=True,
    )
    pose_params = torch.cat(
        [
            torch.tensor(smpl_data['poses']),
            torch.tensor(smpl_data['Rh']),
            torch.tensor(smpl_data['Th']),
        ], dim=-1
    ).float()
    return (
        vertices.squeeze(0),
        joints.squeeze(0),
        joints_transform.squeeze(0),
        pose_params.squeeze(0),
        bones_tranform.squeeze(0),
    )


def cli(args):
    base_dir = args.base_dir
    smpl_dir = args.smpl_dir
    start_idx = args.start_idx
    print (f"processing subject: {base_dir}")
    print (f"SMPL params: {smpl_dir}")
    # smpl body model
    body_model = SMPLlayer(
        model_path=args.smpl_model_path, gender="neutral", 
    )

    # parsing frame ids
    meta_fp = os.path.join(
        os.path.join(base_dir, "annots.npy")
    )
    meta_data = np.load(meta_fp, allow_pickle=True).item()
    frame_ids = range(min(len(meta_data['ims']), args.num_frames))

    # rest state info
    rest_verts, rest_joints, rest_tfs, rest_tfs_bone = load_rest_pose_info(smpl_dir, body_model, start_idx)
    lbs_weights = body_model.weights.float()

    # pose state info
    verts, joints, tfs, params, tf_bones = [], [], [], [], []
    for frame_id in tqdm.tqdm(frame_ids):
        frame_id = frame_id + start_idx
        try:
            _verts, _joints, _tfs, _params, _tfs_bone = (
                load_pose_info(smpl_dir, frame_id, body_model)
            )
            verts.append(_verts)
            joints.append(_joints)
            tfs.append(_tfs)
            params.append(_params)
            tf_bones.append(_tfs_bone)
        except ValueError as e:
            print(e, end=". ")
            print("Append None objects")

            verts.append(None)
            joints.append(None)
            tfs.append(None)
            params.append(None)
            tf_bones.append(None)

    # verts = torch.stack(verts)
    # joints = torch.stack(joints)
    # tfs = torch.stack(tfs)
    # params = torch.stack(params)
    # tf_bones = torch.stack(tf_bones)

    data = {
        "lbs_weights": lbs_weights,  # [6890, 24]
        "rest_verts": rest_verts,  # [6890, 3]
        "rest_joints": rest_joints,  # [24, 3]
        "rest_tfs": rest_tfs,  # [24, 4, 4]
        # "rest_tfs_bone": rest_tfs_bone, # [24, 4, 4]
        "verts": verts,  # [1470, 6890, 3]
        "joints": joints,  # [1470, 24, 3]
        "tfs": tfs,  # [1470, 24, 4, 4]
        # "tf_bones": tf_bones,  # [1470, 24, 4, 4]
        "params": params,  # [1470, 72 + 3 + 3]
        "faces": body_model.faces_tensor,  # [13776, 3]
    }
    save_path = os.path.join(
        base_dir, f"{args.save_file_name}.pt"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print("saving to", save_path)
    torch.save(data, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str)
    parser.add_argument("--smpl_dir", type=str)
    parser.add_argument("--smpl_model_path", type=str, default='data/SMPL_NEUTRAL.pkl')
    parser.add_argument("--save_file_name", type=str, default="pose_data")
    parser.add_argument("--num_frames", type=int, default=np.inf)
    parser.add_argument("--start_idx", type=int, default=0)
    args = parser.parse_args()
    cli(args)
