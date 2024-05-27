import collections
import tqdm
import numpy as np
import torch

MetaData = collections.namedtuple(
    "MetaData", ("lbs_weights",
            "rest_verts",
            "rest_joints",
            "rest_tfs",
            "rest_tfs_bone",
            "verts",
            "joints",
            "tfs",
            "tf_bones",
            "params",
            "faces",)
)

def _work_fn(args):
    index_list, preprocess, fetch_data, process_id = args
    if process_id == 0:
        return [preprocess(fetch_data(i)) for i in tqdm.tqdm(index_list, desc=f"Preprocessing {process_id}")]
    else:
        return [preprocess(fetch_data(i)) for i in index_list]

def get_projection_matrix(cameras, cam_near_far):
    fovx = focal_length_to_fov(cameras.intrins[0, 0], cameras.width)
    fovy = focal_length_to_fov(cameras.intrins[1, 1], cameras.height)

    principal_offset_x = (cameras.intrins[0, 2] - cameras.width / 2) / cameras.width * 2
    principal_offset_y = (cameras.intrins[1, 2] - cameras.height / 2) / cameras.height * 2
    proj = perspective_non_rect(fovx, fovy, principal_offset_x, principal_offset_y, cam_near_far[0], cam_near_far[1])

    return proj

def focal_length_to_fov(focal_length, sensor_length):
    return 2 * np.arctan(0.5 * sensor_length / focal_length)

def perspective_non_rect(fovx=0.7854, fovy=0.7854, principal_offset_x=0, principal_offset_y=0, n=0.1, f=1000.0, device=None):
    y = np.tan(fovy / 2)
    x = np.tan(fovx / 2)
    return torch.tensor([[1/x,    0,            -principal_offset_x,              0], 
                         [           0, 1/-y,            -principal_offset_y,              0], 
                         [           0,    0, -(f+n)/(f-n), -(2*f*n)/(f-n)], 
                         [           0,    0,           -1,              0]], dtype=torch.float32, device=device)
