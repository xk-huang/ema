import os
import os.path as osp
import cv2
import imageio.v2 as imageio
import torch
import numpy as np
import tqdm
import logging

from utils.camera import transform_cameras
from utils.structures import Cameras
from .dataset_utils import MetaData, get_projection_matrix, check_to_log, get_bounds, get_mask_at_box, srgb_to_rgb, get_transforms_t_pose2vitruvian_pose
from .constants import SMPL_CONSTANTS, ZJU_MOCAP_CONSTANTS

LOGGER = logging.getLogger(__name__)


# [FIXME](xk): have bugs in camera pose


class DatasetZJUMoCapAIST(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        aist_pose_pkl_path=None,
        root_dir,
        split,
        mask_dir_name,
        tfs_type,
        views=None,
        begin_frame_id=0,
        num_frames=None,
        frame_interval=None,
        dilate_erode_mask=False,
        order_camera_first=True,
        resize_factor=1.0,
        opencv_camera=True,
        pre_load=False,
        force_reload=False,
        no_image_inputs=False,
        cam_near_far=[0.1, 1000.0],
        width=1024,
        height=1024,
        use_vitruvian_pose=False,
        mask_at_box_bound_pad=0.0,
    ):
        self.parser = SubjectParser(root_dir, mask_dir_name)

        self._prepare_variables(split, views, dilate_erode_mask, resize_factor, opencv_camera, no_image_inputs, cam_near_far, width, height, pre_load, tfs_type, mask_at_box_bound_pad)

        self.index_list = self.dataset_index_list(
            begin_frame_id, 
            num_frames, 
            frame_interval,
            order_camera_first
        )
        self.meta_data: MetaData = self.parser.load_meta_data()

        if use_vitruvian_pose:
            print("Using vitruvian pose")
            _rest_joints = self.meta_data["rest_joints"]

            _tfs_t_pose2vitruvian_pose = get_transforms_t_pose2vitruvian_pose(_rest_joints) 
            self._to_vitruvian_tfs = lambda tfs: torch.matmul(tfs, torch.inverse(_tfs_t_pose2vitruvian_pose))
            self.meta_data["rest_tfs"] = _tfs_t_pose2vitruvian_pose

            _rest_joints = torch.nn.functional.pad(_rest_joints, (0, 1), value=1.0)
            _rest_joints = torch.einsum("ij,ikj->ik", _rest_joints, _tfs_t_pose2vitruvian_pose)
            self.meta_data["rest_joints"] = _rest_joints[..., :-1]

            _verts = torch.nn.functional.pad(self.meta_data["rest_verts"], (0, 1), value=1.0)
            _T = torch.einsum("ij,jkl->ikl", self.meta_data["lbs_weights"], _tfs_t_pose2vitruvian_pose) 
            _verts = torch.einsum("ij,ikj->ik", _verts, _T)
            self.meta_data["rest_verts"] = _verts[..., :-1]
        else:
            print("Using original pose")
            self._to_vitruvian_tfs = lambda x: x

        if pre_load:
            self._pre_load_data(root_dir, split, force_reload, "origin")

        if aist_pose_pkl_path is not None:
            print("Using external pose at {}".format(aist_pose_pkl_path))
            import pickle
            from utils.transforms import axis_angle_to_matrix
            from scripts.utils.lbs import batch_rigid_transform
            from scripts.utils.body_model import to_np, to_tensor


            file_path = aist_pose_pkl_path
            assert os.path.exists(file_path), f'File {file_path} does not exist!'
            with open(file_path, 'rb') as f:
                data = pickle.load(f)


            smpl_poses = data['smpl_poses']  # (N, ???)
            batch_size = len(smpl_poses)
            smpl_poses = torch.from_numpy(smpl_poses[:, :72]).float().view(-1, 24, 3)
            external_global_orient_ = smpl_poses[:, :1]  # (batch_size, 3, 3)
            smpl_poses[:, :1] = 0
            smpl_poses = axis_angle_to_matrix(smpl_poses)
            external_global_orient_ = axis_angle_to_matrix(external_global_orient_)
            external_global_orient = torch.eye(4)[None, None].repeat(batch_size, 1, 1, 1).float()
            external_global_orient[:, :, :3, :3] = external_global_orient_

            SMPL_PATH = 'data/SMPL_NEUTRAL.pkl'
            with open(SMPL_PATH, 'rb') as smpl_file:
                smpl_data = pickle.load(smpl_file, encoding='latin1')

            parents = to_tensor(to_np(smpl_data['kintree_table'][0])).long()
            parents[0] = -1

            _, smpl_poses, _ = batch_rigid_transform(smpl_poses, self.meta_data["rest_joints"].unsqueeze(0).expand(batch_size, -1, -1), parents)
            smpl_poses = torch.matmul(external_global_orient, smpl_poses)
            self.smpl_poses = smpl_poses
            # smpl_poses_4x4 = torch.ones((smpl_poses.shape[0], 24, 4, 4)).float()
            # smpl_poses_4x4[:, :, :3, :3] = smpl_poses
            # self.smpl_poses = smpl_poses_4x4

            self.smpl_trans = data['smpl_trans']  # (N, 3)
            self.smpl_trans = torch.from_numpy(self.smpl_trans).float()
            self.smpl_poses[:, :, :3, 3] += self.smpl_trans[:, None, :] / 100

            self.smpl_scaling = data['smpl_scaling']  # (1,)

            # views & index_list
            views = self.views
            index_list = []
            for camera_id in views:
                index_list.extend([(frame_id, camera_id) for frame_id in range(min(num_frames, self.smpl_trans.shape[0]))])
            self.index_list = index_list
        else:
            print("Using original pose")
        self.aist_pose_pkl_path = aist_pose_pkl_path

    def dataset_index_list(
            self,
            begin_frame_id, 
            num_frames, 
            frame_interval,
            order_camera_first,
        ):
        """_summary_

        Args:
            parser (_type_): _description_
            split (_type_): _description_
            camera_first (bool, optional): _description_. Defaults to True.

        Returns:
            List[Tuple[int, int]]: each image id: (frame_id, camera_id)
        """        
        if check_to_log():
            print = LOGGER.info
        else:
            print = LOGGER.debug

        if num_frames is None:
            num_frames = self.parser.num_frames
            print(f"num_frames is None, set to {num_frames}.")
        if frame_interval is None:
            frame_interval = self.parser.num_frames // num_frames
            print(f"frame_interval is None, set to {self.parser.num_frames} // {num_frames} = {frame_interval}.")

        if frame_interval < 1:
            raise ValueError("frame_interval must be >= 1")

        end_frame_bound = min(self.parser.num_frames, begin_frame_id + num_frames * frame_interval)
        frame_list = list(range(begin_frame_id, end_frame_bound, frame_interval))
        views = self.views

        print(f"camera views: ({len(views)}): {views}")
        print(f"\tcamera e.g., {self.parser.image_files[0][views]}")

        print(f"from begin_frame_id:{begin_frame_id} to end_frame_bound: {end_frame_bound}, interval: {frame_interval}")
        print(f"frame_list ({len(frame_list)}): {frame_list[:5]}...{frame_list[-5:]}")

        print(f"order_camera_first: {order_camera_first}")
        index_list = []
        if order_camera_first:
            for frame_id in frame_list:
                index_list.extend([(frame_id, camera_id) for camera_id in views])
        else:
            for camera_id in views:
                index_list.extend([(frame_id, camera_id) for frame_id in frame_list])
        return index_list

    def _prepare_variables(self, split, views, dilate_erode_mask, resize_factor, opencv_camera, no_image_inputs, cam_near_far, width, height, pre_load, tfs_type, mask_at_box_bound_pad):
        if check_to_log():
            print = LOGGER.info
        else:
            print = LOGGER.debug

        print(f"Loading: {self.__class__.__name__}")
        print(f"split: {split}")
        print(f"dilate_erode_mask: {dilate_erode_mask}")

        self.split = split
        self.resize_factor = resize_factor
        self.dtype = torch.get_default_dtype()
        self.correct_mat = torch.eye(4, dtype=self.dtype)
        if opencv_camera:
            self.correct_mat[[1,2], [1,2]] = -1

        self.cam_near_far = cam_near_far
        if views is None:
            self.views = self.parser.camera_ids
        else:
            if not all([0 <= v < self.parser.num_cameras for v in views]):
                raise ValueError(f"Invalid view id: {views}")
            self.views = views

        self.no_image_inputs = no_image_inputs 
        self.height = height
        self.width = width
        self.dilate_erode_mask = dilate_erode_mask

        # [Note] remove repeated xfms in `tfs_bones`.
        bone_ids_to_uniq_tf_ids = torch.tensor(SMPL_CONSTANTS.BONE_IDS_TO_UNIQ_TF_IDS).long()
        uniq_tf_selector_for_bone = []
        for uniq_tf_ids in torch.unique(bone_ids_to_uniq_tf_ids):
            uniq_tf_selector_for_bone.append(torch.where((uniq_tf_ids == bone_ids_to_uniq_tf_ids) > 0)[0][0])  # select the first tfs among the repeated ones
        self.uniq_tf_selector_for_bone = torch.tensor(uniq_tf_selector_for_bone).long()

        self.pre_load = pre_load

        self.tfs_type = tfs_type

        self.mask_at_box_bound_pad = mask_at_box_bound_pad
        print(f"mask_at_box_bound_pad: {mask_at_box_bound_pad}")

    def _pre_load_data(self, root_dir, split, force_reload, suffix):
        if check_to_log():
            print = LOGGER.info
        else:
            print = LOGGER.debug

        pre_load_pt_path = osp.join(root_dir, f'pre_load.{suffix}.{split}.pt')
        if not os.path.isfile(pre_load_pt_path) or force_reload:
            if force_reload:
                print(f"Force reload {pre_load_pt_path}")

            self.pre_loaded_data = []
            for i in tqdm.trange(len(self), desc="Preloading {} images in {}".format(split, root_dir)):
                self.pre_loaded_data.append(self.preprocess(self.fetch_data(i)))
            print("Caching preloaded {} images".format(len(self)))

            torch.save(self.pre_loaded_data, pre_load_pt_path)
            print("Saved preloaded data to {}".format(pre_load_pt_path))
        else:
            print("Loading preloaded data from {}...".format(pre_load_pt_path))
            self.pre_loaded_data = torch.load(pre_load_pt_path)

            if len(self) != len(self.pre_loaded_data):
                raise ValueError("preloaded data length {} != index_list length {}".format(len(self.pre_loaded_data), len(self)))

            print("Loaded, length {}".format(pre_load_pt_path, len(self.pre_loaded_data)))

    def __len__(self):
        return len(self.index_list)
    
    def __getitem__(self, index):
        if self.pre_load:
            data = self.pre_loaded_data[index]
        else:
            data = self.fetch_data(index)
            data = self.preprocess(data)

        return data

    def fetch_data(self, index):
        # load data
        # XXX
        camera_index = index % len(self.index_list)
        frame_id, camera_id = self.index_list[camera_index]
        K = self.parser.cameras[camera_id]["K"].copy()  # (3, 3)
        w2c = self.parser.cameras[camera_id]["w2c"].copy()  # (4, 4)
        D = self.parser.cameras[camera_id]["D"].copy()  # (5, 1)

        # create pixels
        if self.no_image_inputs:
            rgba = None
        else:
            rgba = np.concatenate(
                [
                    self.parser.load_image(frame_id, camera_id),
                    self.parser.load_mask(frame_id, camera_id, trimap=self.dilate_erode_mask)[
                        ..., None
                    ],
                ],
                axis=-1,
            )   # (H, W, 4)
            rgba = (
                torch.from_numpy(
                    cv2.resize(
                        cv2.undistort(rgba, K, D),
                        (0, 0),
                        fx=self.resize_factor,
                        fy=self.resize_factor,
                        interpolation=cv2.INTER_AREA,
                    )
                ).to(self.dtype)
                / 255.0
            )  # (H * resize_factor, W * resize_factor, 4)

            # LDR image from ZJU-Mocap
            rgba[..., :3] = srgb_to_rgb(rgba[..., :3])

        # create camera
        cameras = Cameras(
            intrins=torch.from_numpy(K).to(self.dtype),
            extrins=torch.from_numpy(w2c).to(self.dtype),
            distorts=None,  # [NOTE] already undistort images
            width=self.width,
            height=self.height,
        )
        cameras = transform_cameras(cameras, self.resize_factor)

        if self.aist_pose_pkl_path is None:
            # skeleton transformation
            tfs = self.meta_data["tfs"][frame_id].type(self.dtype)  # (24, 4, 4)
            # [XXX] the tfs_bone in TAVA is global transforamtion of joints, 
            # not relative one like `tfs` (relative to t-pose), so we need to convert it to relative one.
            # Inconsistent names. :(
        else:
            tfs = self.smpl_poses[index % len(self.smpl_poses)]
            # tfs[1:] = self.meta_data["rest_tfs"][1:]
            tfs[1:] = torch.matmul(torch.inverse(tfs[:1]), tfs[1:])

            tfs[0] = self.meta_data["tfs"][frame_id].type(self.dtype)[0]
            tfs[1:] = torch.matmul(tfs[:1], tfs[1:])
            # mvp transformation
        global_model_transformation = tfs[0]  # (4, 4)

        view_transformation = torch.matmul(self.correct_mat, cameras.extrins)  # (4, 4)
        projection_transformation = get_projection_matrix(cameras, self.cam_near_far)  # (4, 4)

        mv = torch.matmul(view_transformation, global_model_transformation)
        mvp = torch.matmul(projection_transformation, mv)
        vp = torch.matmul(projection_transformation, view_transformation)
        campos = torch.linalg.inv(mv)[:3, 3]  # (3,)

        inv_global_model_transformation = torch.linalg.inv(global_model_transformation)  # (4, 4)
        tfs_in_canon = inv_global_model_transformation @ tfs  # (24, 4, 4)
        tfs_in_canon = self._to_vitruvian_tfs(tfs_in_canon)
        # [XXX] fix tfs_bone. since it's bone transformation, 
        # we remove the root transformation at the beginning.
        tfs_bone_in_canon = tfs_in_canon[SMPL_CONSTANTS.BONE_HEAD_JOINT_IDS]  # (23, 4, 4)
        tfs_bone_in_canon = tfs_bone_in_canon[self.uniq_tf_selector_for_bone]  # (num_uniq_tfs_bone, 4, 4)

        params = self.meta_data["params"][frame_id].type(self.dtype)

        verts_in_model = self.meta_data["verts"][frame_id].type(self.dtype)  #  (6889, 3)

        bounds_in_model = get_bounds(verts_in_model, self.mask_at_box_bound_pad)  # (2, 3)
        H, W = cameras.height, cameras.width
        K, R, T = cameras.intrins, cameras.extrins[:3, :3], cameras.extrins[:3, 3:]
        mask_at_box = get_mask_at_box(H, W, K, R, T, bounds_in_model)  # (H, W), bool

        return {
            "camera_id": camera_id,  # int
            "frame_id": frame_id,  # int
            # transformations
            "mv": mv[None],  # (1, 4, 4)
            "mvp": mvp[None], # (1, 4, 4)
            "vp": vp[None], # (1, 4, 4)
            # "projection_transformation": projection_transformation[None], # (1, 4, 4)
            # "view_transformation": view_transformation[None],  # (1, 4, 4)
            # "global_model_transformation": global_model_transformation[None], # (1, 4, 4)
            "tfs_in_canon": tfs_in_canon[None],  # (1, 24, 4, 4)
            "tfs_bone_in_canon": tfs_bone_in_canon[None],  # (1, num_uniq_tfs_bone, 4, 4)
            "params": params[None],  # (1, 78)
            "verts_in_model": verts_in_model[None],  # (1, 6890, 3)
            "mask_at_box": mask_at_box[None],  # (1, H, W)
            # nvdiffrec
            "campos": campos[None], # (1, 3)
            "resolution": [cameras.height, cameras.width], # List[int, int]
            "img": rgba[None] if rgba is not None else rgba,  # (1, H, W, 4)
        }

    EXCEPT_KEYS= ["subject_id", "camera_id", "frame_id", "resolution", "spp", "img"]
    def collate(self, batch):
        collate_batch = {
            'resolution': batch[0]['resolution'],
            # 'spp': batch[0]['spp'],
            'img' : torch.cat(list([item['img'] for item in batch]), dim=0) if not self.no_image_inputs else None,
            'camera_id': [item['camera_id'] for item in batch],  # List[int]
            'frame_id': [item['frame_id'] for item in batch],  # List[int]
        }

        for k in batch[0].keys():
            if k in self.EXCEPT_KEYS:
                continue
            collate_batch[k] = torch.cat([item[k] for item in batch], dim=0)

        return collate_batch 

    def preprocess(self, data):
        return data


    def get_animation_meta_data(self):
        # load canonical meta info.
        # rest_matrixs = self.meta_data["rest_tfs_bone"][1:]  # [23, 4, 4]
        bone_tails = self.meta_data["rest_joints"][
            [
                SMPL_CONSTANTS.JOINT_NAMES.index(tail_name)
                for _, tail_name in SMPL_CONSTANTS.BONE_NAMES
            ]
        ]  # [23, 3]
        bone_heads = self.meta_data["rest_joints"][
            [
                SMPL_CONSTANTS.JOINT_NAMES.index(head_name)
                for head_name, _ in SMPL_CONSTANTS.BONE_NAMES
            ]
        ]  # [23, 3]

        # bones_rest = Bones(
        #     heads=rest_heads,
        #     tails=rest_tails.type(self.dtype),
        #     transforms=None,
        # )  # real bones [23,]
        # bone_heads, bone_tails = get_end_points(bones_rest)  # [23, 3]

        if self.tfs_type == "joint":
            bone2uniq_tf = torch.tensor(SMPL_CONSTANTS.BONE_HEAD_JOINT_IDS).long()
            num_uniq_tfs = self.meta_data["rest_joints"].shape[0]
        elif self.tfs_type == "bone":
            bone2uniq_tf = torch.tensor(SMPL_CONSTANTS.BONE_IDS_TO_UNIQ_TF_IDS).long()
            num_uniq_tfs = len(torch.unique(bone2uniq_tf))
        else:
            raise ValueError(f"Unknown tfs_type {self.tfs_type}")
 
        return {
            # base SMPL
            'rest_verts_in_canon': self.meta_data["rest_verts"],  # (6890, 3)
            'lbs_weights': self.meta_data["lbs_weights"],  # (6890, 24)
            'rest_joints_in_canon': self.meta_data["rest_joints"],  # (24, 3)
            'rest_tfs_global': self.meta_data["rest_tfs"],  # (24, 4, 4)
            # 'rest_tfs_bone_global': self.meta_data["rest_tfs_bone"],  # (24, 4, 4)
            'faces': self.meta_data["faces"],  # (13776, 3)
            # "bones_rest": bones_rest,
            "bone_heads": bone_heads,
            "bone_tails": bone_tails,
            "bone2uniq_tf": bone2uniq_tf,
            "num_uniq_tfs": num_uniq_tfs
        }


class SubjectParser():
    def __init__(
            self,
            root_dir,
            mask_dir_name,
        ):
        if check_to_log():
            print = LOGGER.info
        else:
            print = LOGGER.debug

        self.root_dir = root_dir

        self.image_dir = osp.join(root_dir, 'images')
        if not osp.exists(self.image_dir):
            # [NOTE] Original ZJU-Mocap is different from Eacy Mocap structure
            self.image_dir = self.root_dir

        if mask_dir_name not in ZJU_MOCAP_CONSTANTS.MASK_DIR_NAME_TO_MASK_SUFFIX:
            raise ValueError('mask_dir_name should be one of {}'.format(ZJU_MOCAP_CONSTANTS.MASK_DIR_NAME_TO_MASK_SUFFIX.keys()))
        self.mask_dir = osp.join(root_dir, mask_dir_name)
        self.mask_suffix = ZJU_MOCAP_CONSTANTS.MASK_DIR_NAME_TO_MASK_SUFFIX[mask_dir_name]

        print('Loading ZJU-MoCap dataset from {}'.format(root_dir))
        print('Loading images from {}'.format(self.image_dir))
        print('Loading masks from {}'.format(self.mask_dir))

        annots_fp = os.path.join(self.root_dir, "annots.npy")
        annots_data = np.load(annots_fp, allow_pickle=True).item()
        self.cameras = self._parse_camera(annots_data)

        # XXX: there might be a bug. we want "0/000000.jpg"
        # but there are "images/0/000000.jpg" sometimes.
        self.image_files = np.array(
            [[self.remove_prefix(fp, "images/") for fp in fps["ims"]] for fps in annots_data["ims"]], dtype=str
        )  # frame X camera
        self._frame_ids = list(range(self.image_files.shape[0]))
        self._camera_ids = list(range(self.image_files.shape[1]))

    @staticmethod
    def remove_prefix(text, prefix):
        return text[text.startswith(prefix) and len(prefix):]

    @property
    def camera_ids(self):
        return self._camera_ids

    @property
    def num_cameras(self):
        return len(self._camera_ids)

    @property
    def frame_ids(self):
        return self._frame_ids

    @property
    def num_frames(self):
        return len(self._frame_ids)
    
    def _parse_camera(self, annots_data):
        K = np.array(annots_data["cams"]["K"]).astype(np.float32)  # (3, 3)
        R = np.array(annots_data["cams"]["R"]).astype(np.float32)  # (3, 3)
        # [NOTE] K, R, D are in M, but T is in mm. So we need to divide 1000.0
        T = np.array(annots_data["cams"]["T"]).astype(np.float32) / 1000.0  # (3, 1)
        D = np.array(annots_data["cams"]["D"]).astype(np.float32)  # (5, 1)
        cameras = {
            cid: {
                "K": K[cid],
                "D": D[cid],
                "w2c": np.concatenate(
                    [
                        np.concatenate([R[cid], T[cid]], axis=-1),
                        np.array([[0.0, 0.0, 0.0, 1.0]], dtype=R.dtype),
                    ],
                    axis=0,
                ),
            }
            for cid in range(K.shape[0])
        }
        return cameras

    def load_image(self, frame_id, camera_id):
        path = os.path.join(
            self.image_dir, self.image_files[frame_id, camera_id]
        )
        image = imageio.imread(path)
        return image  # shape [HEIGHT, WIDTH, 3], value 0 ~ 255

    def load_mask(self, frame_id, camera_id, trimap=True):
        path = os.path.join(
            self.mask_dir,
            self.image_files[frame_id, camera_id].replace(".jpg", self.mask_suffix),
        )
        mask = (imageio.imread(path) != 0).astype(np.uint8) * 255
        if trimap:
            mask = self._process_mask(mask, 5, 128)
        return mask  # shape [HEIGHT, WIDTH], value [0, (128,) 255]

    def _process_mask(self, mask, border: int = 5, ignore_value: int = 128):
        kernel = np.ones((border, border), np.uint8)
        mask_erode = cv2.erode(mask.copy(), kernel)
        mask_dilate = cv2.dilate(mask.copy(), kernel)
        mask[mask_dilate != mask_erode] = ignore_value
        return mask

    def load_meta_data(self, frame_ids=None):
        meta_data_path = os.path.join(self.root_dir, "pose_data.pt")
        if not osp.exists(meta_data_path):
            raise ValueError(
                f"meta data file {meta_data_path} does not exist."
                "run preprocessing first."
            )
        data = torch.load(meta_data_path)

        keys = [
            "lbs_weights",
            "rest_verts",
            "rest_joints",
            "rest_tfs",
            # "rest_tfs_bone",  # Discard, no SNARF here.
            "verts",
            "joints",
            "tfs",
            # "tf_bones",  # Discard, no SNARF here.
            "params",
            "faces",  # [NOTE] for render SMPL mesh
        ]
        return {
            key: (
                data[key][frame_ids]
                if (
                    frame_ids is not None
                    and key in ["verts", "joints", "tfs", "tf_bones", "params"]
                )
                else data[key]
            )
            for key in keys
        }
