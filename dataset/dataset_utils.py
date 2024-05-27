import collections
import tqdm
import numpy as np
import torch
import cv2
import random

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

def check_to_log():
    return torch.distributed.is_initialized() is False or torch.distributed.get_rank() == 0

def get_bounds(xyz, box_padding=0.0):
    min_xyz, _ = torch.min(xyz, dim=0)
    max_xyz, _ = torch.max(xyz, dim=0)
    min_xyz = min_xyz - box_padding
    max_xyz = max_xyz + box_padding
    bounds = torch.stack([min_xyz, max_xyz], dim=0)
    bounds = bounds.type(torch.float32)
    return bounds

def get_mask_at_box(H, W, K, R, T, bounds):
    ray_o, ray_d = get_rays(H, W, K, R, T)
    _, _, mask_at_box = get_near_far(bounds, ray_o, ray_d)
    return mask_at_box

def get_rays(H, W, K, R, T):
    # calculate the camera origin
    rays_o = -torch.matmul(R.T, T).ravel()
    # calculate the world coodinates of pixels
    i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32),
                       torch.arange(H, dtype=torch.float32),
                       indexing='xy')
    xy1 = torch.stack([i, j, torch.ones_like(i)], dim=2)
    pixel_camera = torch.matmul(xy1, torch.inverse(K).T)
    pixel_world = torch.matmul(pixel_camera - T.ravel(), R)
    # calculate the ray direction
    rays_d = pixel_world - rays_o[None, None]
    rays_d = rays_d / torch.norm(rays_d, dim=2, keepdim=True)
    rays_o = torch.broadcast_to(rays_o, rays_d.shape)
    return rays_o, rays_d

def get_near_far(bounds, ray_o, ray_d):
    """calculate intersections with 3d bounding box"""
    spatial_shape = ray_o.shape[:2]
    ray_o, ray_d = ray_o.reshape(-1, 3), ray_d.reshape(-1, 3)

    bounds = bounds + torch.tensor([-0.01, 0.01])[:, None]
    nominator = bounds[None] - ray_o[:, None]
    # calculate the step of intersections at six planes of the 3d bounding box
    d_intersect = (nominator / ray_d[:, None]).reshape(-1, 6)
    # calculate the six interections
    p_intersect = d_intersect[..., None] * ray_d[:, None] + ray_o[:, None]
    # calculate the intersections located at the 3d bounding box
    min_x, min_y, min_z, max_x, max_y, max_z = bounds.ravel()
    eps = 1e-6
    p_mask_at_box = (p_intersect[..., 0] >= (min_x - eps)) * \
                    (p_intersect[..., 0] <= (max_x + eps)) * \
                    (p_intersect[..., 1] >= (min_y - eps)) * \
                    (p_intersect[..., 1] <= (max_y + eps)) * \
                    (p_intersect[..., 2] >= (min_z - eps)) * \
                    (p_intersect[..., 2] <= (max_z + eps))
    # obtain the intersections of rays which intersect exactly twice
    mask_at_box = p_mask_at_box.sum(-1) == 2
    p_intervals = p_intersect[mask_at_box][p_mask_at_box[mask_at_box]].reshape(
        -1, 2, 3)

    # calculate the step of intersections
    ray_o = ray_o[mask_at_box]
    ray_d = ray_d[mask_at_box]
    norm_ray = torch.norm(ray_d, dim=1)
    d0 = torch.norm(p_intervals[:, 0] - ray_o, dim=1) / norm_ray
    d1 = torch.norm(p_intervals[:, 1] - ray_o, dim=1) / norm_ray
    near = np.minimum(d0, d1)
    far = np.maximum(d0, d1)

    return near, far, mask_at_box.reshape(spatial_shape, -1)

def worker_init_fn(worker_id):
    cv2.setNumThreads(1)  # NOTE: OpenCV undistort is why all cores are taken. GBY OpenCV

    # NOTE: offcial guide，https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    # print("worker_seed: ", worker_seed)

    # Community solution: https://github.com/wagnew3/ARM/blob/f55c6b0fac44d9d749e7804d99169a39d30c2111/data_loader.py#L21
    # base_seed = np.random.get_state()[1][0]
    # np.random.seed(worker_id + base_seed)
    # random.seed(worker_id + base_seed)

def _srgb_to_rgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(f <= 0.04045, f / 12.92, torch.pow((torch.clamp(f, 0.04045) + 0.055) / 1.055, 2.4))

def srgb_to_rgb(f: torch.Tensor) -> torch.Tensor:
    assert f.shape[-1] == 3 or f.shape[-1] == 4
    out = torch.cat((_srgb_to_rgb(f[..., 0:3]), f[..., 3:4]), dim=-1) if f.shape[-1] == 4 else _srgb_to_rgb(f)
    assert out.shape[0] == f.shape[0] and out.shape[1] == f.shape[1] and out.shape[2] == f.shape[2]
    return out

def get_transforms_t_pose2vitruvian_pose(rest_joints):
    """
    Args:
        rest_joints: (24, 3) tensor
    Returns:
        bone_transforms_02v: (24, 4, 4) tensor
    """
    device = rest_joints.device

    from scipy.spatial.transform import Rotation as R
    import torch.nn.functional as F

    rot45p = torch.tensor(R.from_euler('z', 45, degrees=True).as_matrix(), dtype=torch.float32, device=device)
    rot45n = torch.tensor(R.from_euler('z', -45, degrees=True).as_matrix(), dtype=torch.float32, device=device)
    # Specify the bone transformations that transform a SMPL A-pose mesh
    # to a star-shaped A-pose (i.e. Vitruvian A-pose)
    bone_transforms_02v = torch.eye(4, dtype=torch.float32, device=device).reshape(1, 4, 4).repeat(24, 1, 1)

    # First chain: L-hip (1), L-knee (4), L-ankle (7), L-foot (10)
    R_02v_l = []
    t_02v_l = []
    chain = [1, 4, 7, 10]
    rot = rot45p
    for i, j_idx in enumerate(chain):
        R_02v_l.append(rot)
        t = rest_joints[j_idx]
        if i > 0:
            parent = chain[i-1]
            t_p = rest_joints[parent]
            t = torch.matmul(rot, t - t_p)
            t = t + t_02v_l[i-1]

        t_02v_l.append(t)

    R_02v_l = torch.stack(R_02v_l, dim=0)
    t_02v_l = torch.stack(t_02v_l, dim=0)
    t_02v_l = t_02v_l - torch.matmul(rest_joints[chain], rot.transpose(0, 1))

    R_02v_l = F.pad(R_02v_l, (0, 0, 0, 1))  # 4 x 4 x 3
    t_02v_l = F.pad(t_02v_l, (0, 1), value=1.0)   # 4 x 4

    bone_transforms_02v[chain] = torch.cat([R_02v_l, t_02v_l.unsqueeze(-1)], dim=-1)

    # Second chain: R-hip (2), R-knee (5), R-ankle (8), R-foot (11)
    R_02v_r = []
    t_02v_r = []
    chain = [2, 5, 8, 11]
    rot = rot45n
    for i, j_idx in enumerate(chain):
        # bone_transforms_02v[j_idx, :3, :3] = rot
        R_02v_r.append(rot)
        t = rest_joints[j_idx]
        if i > 0:
            parent = chain[i-1]
            t_p = rest_joints[parent]
            t = torch.matmul(rot, t - t_p)
            t = t + t_02v_r[i-1]

        t_02v_r.append(t)

    # bone_transforms_02v[chain, :3, -1] -= np.dot(Jtr[chain], rot.T)
    R_02v_r = torch.stack(R_02v_r, dim=0)
    t_02v_r = torch.stack(t_02v_r, dim=0)
    t_02v_r = t_02v_r - torch.matmul(rest_joints[chain], rot.transpose(0, 1))

    R_02v_r = F.pad(R_02v_r, (0, 0, 0, 1))  # 4 x 3
    t_02v_r = F.pad(t_02v_r, (0, 1), value=1.0)   # 4 x 4

    bone_transforms_02v[chain] = torch.cat([R_02v_r, t_02v_r.unsqueeze(-1)], dim=-1)

    return bone_transforms_02v
