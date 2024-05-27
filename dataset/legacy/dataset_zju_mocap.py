import os
import os.path as osp
import glob
import json
import cv2
import imageio.v2 as imageio
import torch
import numpy as np
import json
import argparse
import collections
import pickle
import tqdm
import logging

from utils.camera import transform_cameras
from utils.structures import Cameras, Bones
from utils.bones import get_end_points
from render import util
from .dataset_utils import MetaData, get_projection_matrix

LOGGER = logging.getLogger(__name__)


class DatasetZJUMocap(torch.utils.data.Dataset):
    VALID_SPLIT_NAME = ['all', 'train', 'val_ind', 'val_ood', 'val_view', 'test', 'dev_run']
    DEFAULT_MASK_DIR_NAME = 'mask_cihp'
    def __init__(self, FLAGS, split, examples=None):
        if split not in self.VALID_SPLIT_NAME:
            raise ValueError('split name should be one of {}'.format(self.VALID_SPLIT_NAME))

        self.FLAGS = FLAGS
        if self.FLAGS.local_rank == 0:
            print = LOGGER.info
        else:
            print = LOGGER.debug
        self.split = split
        self.examples = examples
        self.is_training = (split in ["train", "all", "dev_run"])
        self.dilate_erode_mask = not self.is_training
        mask_dir_name = FLAGS.mask_dir_name if self.is_training else self.DEFAULT_MASK_DIR_NAME

        print(f"split: {self.split}, is_training: {self.is_training}, dilate_erode_mask: {self.dilate_erode_mask}, mask_dir_name: {mask_dir_name}")
        self.parser = SubjectParser(subject_id=FLAGS.subject_id, root_fp=FLAGS.root_fp, mask_dir_name=mask_dir_name)

        if self.FLAGS.dev_run and self.split == "dev_run":
            print(f"Running in dev mode with {self.FLAGS.dev_run_num_frames} frames.")
            self.index_list = self.create_dev_run_split(dev_run_num_frames=self.FLAGS.dev_run_num_frames)
        else:
            self.index_list = self.dataset_index_list(self.parser, self.split, FLAGS.camera_first, views=FLAGS.views)
        full_n_images = len(self.index_list)
        
        if split in ["val_ind", "val_ood", "val_view", "test"] and examples is not None:
            render_every = len(self.index_list) // examples
            self.index_list = self.index_list[::render_every]
            print(f"render every {render_every} frames, all {examples} frames will be rendered.")

        self.n_images = len(self.index_list)
        self.resize_factor = FLAGS.resize_factor
        # self.color_bkgd_aug = FLAGS.color_bkgd_aug
        self.dtype = torch.get_default_dtype()

        self.meta_data: MetaData = self.parser.load_meta_data()

        self.correct_mat = torch.eye(4, dtype=self.dtype)
        if FLAGS.opencv_camera:
            self.correct_mat[[1,2], [1,2]] = -1

        if examples is not None and examples < self.n_images:
            if split != 'dev_run':
                print(f"Ops! Using {examples} examples, instead of full {full_n_images}. (maybe too much images to load or not enough iteration)")
                self.n_images = examples
        else:
            print(f"Using full {self.n_images} examples when iteration")

        # [Note] remove repeated xfms in `tfs_bones`.
        bone_ids_to_uniq_tf_ids = torch.tensor(self.parser.BONE_IDS_TO_UNIQ_TF_IDS).long()
        uniq_tf_selector_for_bone = []
        for uniq_tf_ids in torch.unique(bone_ids_to_uniq_tf_ids):
            uniq_tf_selector_for_bone.append(torch.where((uniq_tf_ids == bone_ids_to_uniq_tf_ids) > 0)[0][0])  # select the first tfs among the repeated ones
        self.uniq_tf_selector_for_bone = torch.tensor(uniq_tf_selector_for_bone).long()

        # Pre-load from disc to avoid slow png parsing
        if self.FLAGS.pre_load:
            pre_load_pt = osp.join(self.parser.root_dir, f"pre_load.{self.split}.pt")
            if not os.path.isfile(pre_load_pt) or self.FLAGS.force_reload:
                if self.FLAGS.force_reload:
                    print(f"Force reload {pre_load_pt}")

                self.preloaded_data = []
                for i in tqdm.trange(self.n_images, desc="Preloading {} images in {}".format(self.split, self.parser.root_dir)):
                    self.preloaded_data.append(self.preprocess(self.fetch_data(i)))
                print("[INFO] Caching preloaded {} images".format(self.n_images))

                torch.save(self.preloaded_data, pre_load_pt)
                print("[INFO] Saved preloaded data to {}".format(pre_load_pt))
            else:
                self.preloaded_data = torch.load(pre_load_pt)
                print("[INFO] Loading preloaded data from {}, length {}".format(pre_load_pt, len(self.preloaded_data)))
        
        print(f"no_image_inputs: {FLAGS.no_image_inputs}")
        print(f"The length of the Dataset object: {len(self)}")

    EXCEPT_KEYS= ["subject_id", "camera_id", "frame_id", "resolution", "spp", "img"]

    def collate(self, batch):
        collate_batch = {
            'resolution': batch[0]['resolution'],
            # 'spp': batch[0]['spp'],
            'img' : torch.cat(list([item['img'] for item in batch]), dim=0) if not self.FLAGS.no_image_inputs else None,
            'subject_id': [item['subject_id'] for item in batch],  # List[int]
            'camera_id': [item['camera_id'] for item in batch],  # List[int]
            'frame_id': [item['frame_id'] for item in batch],  # List[int]
        }

        for k in batch[0].keys():
            if k in self.EXCEPT_KEYS:
                continue
            collate_batch[k] = torch.cat(list([item[k] for item in batch]), dim=0)

        return collate_batch 

    def get_animation_meta_data(self):
        # load canonical meta info.
        rest_matrixs = self.meta_data["rest_tfs_bone"][1:]  # [23, 4, 4]
        rest_tails = self.meta_data["rest_joints"][
            [
                self.parser.JOINT_NAMES.index(tail_name)
                for _, tail_name in self.parser.BONE_NAMES
            ]
        ]  # [23, 3]
        bones_rest = Bones(
            heads=None,
            tails=torch.from_numpy(rest_tails).to(self.dtype),
            transforms=torch.from_numpy(rest_matrixs).to(self.dtype),
        )  # real bones [23,]
        bone_heads, bone_tails = get_end_points(bones_rest)  # [23, 3]
        if self.FLAGS.tfs_type == "joint":
            bone2uniq_tf = torch.tensor(self.parser.BONE_HEAD_JOINT_IDS).long()
            num_uniq_tfs = self.meta_data["rest_joints"].shape[0]
        elif self.FLAGS.tfs_type == "bone":
            bone2uniq_tf = torch.tensor(self.parser.BONE_IDS_TO_UNIQ_TF_IDS).long()
            num_uniq_tfs = len(torch.unique(bone2uniq_tf))
        else:
            raise ValueError(f"Unknown tfs_type {self.FLAGS.tfs_type}")
 
        return {
            # base SMPL
            'rest_verts_in_canon': torch.from_numpy(self.meta_data["rest_verts"]),  # (6890, 3)
            'lbs_weights': torch.from_numpy(self.meta_data["lbs_weights"]),  # (6890, 24)
            'rest_joints_in_canon': torch.from_numpy(self.meta_data["rest_joints"]),  # (24, 3)
            # 'rest_tfs_global': torch.from_numpy(self.meta_data["rest_tfs"]),  # (24, 4, 4)
            # 'rest_tfs_bone_global': torch.from_numpy(self.meta_data["rest_tfs_bone"]),  # (24, 4, 4)
            'faces': torch.from_numpy(self.meta_data["faces"]),  # (13776, 3)
            # "bones_rest": bones_rest,
            "bone_heads": bone_heads,
            "bone_tails": bone_tails,
            "bone2uniq_tf": bone2uniq_tf,
            "num_uniq_tfs": num_uniq_tfs
        }

    def preprocess(self, data):
        # rgba = data["img"]
        # if rgba is None:
        #     return data

        # image, alpha = torch.split(rgba, [3, 1], dim=-1)

        # if self.training:
        #     if self.color_bkgd_aug == "random":
        #         color_bkgd = torch.rand(3, dtype=rgba.dtype)
        #     elif self.color_bkgd_aug == "white":
        #         color_bkgd = torch.ones(3, dtype=rgba.dtype)
        #     elif self.color_bkgd_aug == "black":
        #         color_bkgd = torch.zeros(3, dtype=rgba.dtype)
        #     elif self.color_bkgd_aug == 'gray':
        #         color_bkgd = torch.ones(3, dtype=rgba.dtype) * 0.5
        # else:
        #     # just use black during inference
        #     color_bkgd = torch.zeros(3, dtype=rgba.dtype)

        # image = image * (alpha != 0) + color_bkgd * (alpha == 0)
        # rgba = torch.cat([image, alpha], dim=-1)
        # data["img"] = rgba
        return data

    def fetch_data(self, index):
        """Fetch the data (it maybe cached for multiple batches)."""
        # load data
        frame_id, camera_id = self.index_list[index]
        K = self.parser.cameras[camera_id]["K"].copy()  # (3, 3)
        w2c = self.parser.cameras[camera_id]["w2c"].copy()  # (4, 4)
        D = self.parser.cameras[camera_id]["D"].copy()  # (5, 1)

        # create pixels
        if self.FLAGS.no_image_inputs:
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
            rgba[..., :3] = util.srgb_to_rgb(rgba[..., :3])

        # create rays from camera
        cameras = Cameras(
            intrins=torch.from_numpy(K).to(self.dtype),
            extrins=torch.from_numpy(w2c).to(self.dtype),
            distorts=None,
            width=self.parser.WIDTH,
            height=self.parser.HEIGHT,
        )
        cameras = transform_cameras(cameras, self.resize_factor)

        projection_transformation = get_projection_matrix(cameras, self.FLAGS.cam_near_far)  # (4, 4)
        view_transformation = cameras.extrins
        view_transformation = torch.matmul(self.correct_mat, view_transformation)  # (4, 4)

        tfs = torch.tensor(self.meta_data["tfs"][frame_id], dtype=self.dtype)  # (24, 4, 4)
        # [XXX] the tfs_bone in TAVA is global transforamtion of joinst, 
        # not relative one like `tfs`, so we need to convert it to relative one.
        # Inconsistent names. :(
        tfs_bone = torch.tensor(self.meta_data["tf_bones"][frame_id], dtype=self.dtype)  # (24, 4, 4)
        verts_in_model = torch.tensor(self.meta_data["verts"][frame_id], dtype=self.dtype)  #  (6890, 3)
        global_model_transformation = tfs_bone[0]  # (4, 4)

        mv = torch.matmul(view_transformation, global_model_transformation)
        mvp = torch.matmul(projection_transformation, mv)
        vp = torch.matmul(projection_transformation, view_transformation)
        campos = torch.linalg.inv(mv)[:3, 3]  # (3,)

        inv_global_model_transformation = torch.linalg.inv(global_model_transformation)  # (4, 4)
        tfs_in_canon = inv_global_model_transformation @ tfs  # (24, 4, 4)
        # [XXX] fix tfs_bone. since it's bone transformation, 
        # we remove the root transformation at the beginning.
        tfs_bone_in_canon = tfs_in_canon[self.parser.BONE_HEAD_JOINT_IDS]  # (23, 4, 4)
        tfs_bone_in_canon = tfs_bone_in_canon[self.uniq_tf_selector_for_bone]  # (num_uniq_tfs_bone, 4, 4)

        return {
            "subject_id": self.parser.subject_id, # int
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
            "params": torch.tensor(self.meta_data["params"][frame_id:frame_id+1], dtype=self.dtype),  # (1, 78)
            "verts_in_model": verts_in_model[None],  # (1, 6890, 3)
            # nvdiffrec
            "campos": campos[None], # (1, 3)
            "resolution": [cameras.height, cameras.width], # List[int, int]
            # "spp": self.FLAGS.spp,  # int
            "img": rgba[None] if rgba is not None else rgba,  # (1, H, W, 4)
        }

    def __len__(self):
        return self.n_images if self.examples is None else self.examples

    def __getitem__(self, index):
        # [XXX] may leads to infinite loop when iterate over the Dataset without DataLoader
        # https://discuss.pytorch.org/t/why-does-my-custom-dataset-class-loop-forever/50785/6
        index = index % self.n_images

        if self.FLAGS.pre_load:
            data = self.preloaded_data[index]
        else:
            data = self.fetch_data(index)
            data = self.preprocess(data)
        
        return data

    def create_dev_run_split(self, dev_run_num_frames=1):
        """_summary_

        Args:
            parser (_type_): _description_
            split (_type_): _description_
            camera_first (bool, optional): _description_. Defaults to True.

        Returns:
            List[Tuple[int, int]]: each image id: (frame_id, camera_id)
        """       
        num_frames, num_cameras = self.parser.image_files.shape

        index_list = []
        max_num_frames = min(num_frames, dev_run_num_frames)
        interval = max(1, max_num_frames // dev_run_num_frames)

        for frame_id in range(0, max_num_frames, interval):
            for camera_id in range(num_cameras):
                index_list.append((frame_id, camera_id))

        return index_list

    def dataset_index_list(self, parser, split, camera_first=True, views=None):
        """_summary_

        Args:
            parser (_type_): _description_
            split (_type_): _description_
            camera_first (bool, optional): _description_. Defaults to True.

        Returns:
            List[Tuple[int, int]]: each image id: (frame_id, camera_id)
        """        
        if views is None:
            camera_ids = self._dataset_view_split(parser, split)
        else:
            camera_ids = views
        print(f"camera_ids ({len(camera_ids)}): {camera_ids}")

        frame_list = self._dataset_frame_split(parser, split)
        print(f"frame_list ({len(frame_list)}): {frame_list[:3]}...{frame_list[-3:]}")
        index_list = []
        if camera_first:
            for frame_id in frame_list:
                index_list.extend([(frame_id, camera_id) for camera_id in camera_ids])
        else:
            for camera_id in camera_ids:
                index_list.extend([(frame_id, camera_id) for frame_id in frame_list])
        return index_list

    def _dataset_view_split(self, parser, split):
        _train_camera_ids = [0, 6, 12, 18]
        
        if split == "all":
            camera_ids = parser.camera_ids
        elif split == "train":
            camera_ids = _train_camera_ids
        elif split in ["val_ind", "val_ood", "val_view"]:
            camera_ids = list(set(parser.camera_ids) - set(_train_camera_ids))
        elif split == "test":
            camera_ids = [0]
        return camera_ids

    def _dataset_frame_split(self, parser, split):
        if split in ["train", "val_view"]:
            splits_fp = os.path.join(parser.root_dir, "splits/train.txt")
        else:
            splits_fp = os.path.join(parser.root_dir, f"splits/{split}.txt")
        with open(splits_fp, mode="r") as fp:
            frame_list = np.loadtxt(fp, dtype=int).tolist()
        return frame_list


class SubjectParser:
    """Single subject data parser."""

    WIDTH = 1024
    HEIGHT = 1024

    JOINT_NAMES = [
        "root",
        "lhip",
        "rhip",
        "belly",
        "lknee",
        "rknee",
        "spine",
        "lankle",
        "rankle",
        "chest",
        "ltoes",
        "rtoes",
        "neck",
        "linshoulder",
        "rinshoulder",
        "head",
        "lshoulder",
        "rshoulder",
        "lelbow",
        "relbow",
        "lwrist",
        "rwrist",
        "lhand",
        "rhand",
    ]

    BONE_NAMES = [
        ("root", "lhip"),
        ("root", "rhip"),
        ("root", "belly"),
        ("lhip", "lknee"),
        ("rhip", "rknee"),
        ("belly", "spine"),
        ("lknee", "lankle"),
        ("rknee", "rankle"),
        ("spine", "chest"),
        ("lankle", "ltoes"),
        ("rankle", "rtoes"),
        ("chest", "neck"),
        ("chest", "linshoulder"),
        ("chest", "rinshoulder"),
        ("neck", "head"),
        ("linshoulder", "lshoulder"),
        ("rinshoulder", "rshoulder"),
        ("lshoulder", "lelbow"),
        ("rshoulder", "relbow"),
        ("lelbow", "lwrist"),
        ("relbow", "rwrist"),
        ("lwrist", "lhand"),
        ("rwrist", "rhand"),
    ]

    BONE_IDS_TO_UNIQ_TF_IDS = [
        0,
        0,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        9,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
    ]

    # BONE_HEAD_JOINT_IDS = [JOINT_NAMES.index(head) for (head, _) in BONE_NAMES]
    BONE_HEAD_JOINT_IDS = [
        0,
        0,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        9,
        9,
        12,
        13,
        14,
        16,
        17,
        18,
        19,
        20,
        21
    ]
    MASK_SUFFIX = {
        "mask_cihp": ".png",
        "mask-schp": "_0.png"
    }

    def __init__(self, subject_id: int, root_fp: str, mask_dir_name: str):

        # if not root_fp.startswith("/"):
        #     # allow relative path. e.g., "./data/zju/"
        #     root_fp = os.path.join(
        #         os.path.dirname(os.path.abspath(__file__)), 
        #         "..", "..",
        #         root_fp,
        #     )

        self.subject_id = subject_id
        self.root_fp = root_fp
        self.root_dir = os.path.join(root_fp, "CoreView_%d" % subject_id)
        print(f"root_dir: {self.root_dir}")

        image_dir = os.path.join(self.root_dir, "images")
        if not os.path.exists(image_dir):
            image_dir = self.root_dir
        self.image_dir = image_dir
        self.mask_dir = os.path.join(self.root_dir, mask_dir_name)
        self.mask_suffix = self.MASK_SUFFIX[mask_dir_name]
        self.splits_dir = os.path.join(self.root_dir, "splits")

        print(f"image_dir: {self.image_dir}")
        print(f"mask_dir: {self.mask_dir}")

        annots_fp = os.path.join(self.root_dir, "annots.npy")
        annots_data = np.load(annots_fp, allow_pickle=True).item()
        self.cameras = self._parse_camera(annots_data)

        # [1470 x 21]
        self.image_files = np.array(
            [[fp for fp in fps["ims"]] for fps in annots_data["ims"]], dtype=str
        )
        self._frame_ids = list(range(self.image_files.shape[0]))
        self._camera_ids = list(range(self.image_files.shape[1]))

        # If "splits" subfolder does not exist, run our clustering algorithm to 
        # generate train/val/test splits based on pose similarity, as described
        # in the paper.
        if not os.path.exists(self.splits_dir):
            self._create_splits()

    @property
    def camera_ids(self):
        return self._camera_ids

    @property
    def frame_ids(self):
        return self._frame_ids

    def _parse_camera(self, annots_data):
        K = np.array(annots_data["cams"]["K"]).astype(np.float32)
        R = np.array(annots_data["cams"]["R"]).astype(np.float32)
        T = np.array(annots_data["cams"]["T"]).astype(np.float32) / 1000.0
        D = np.array(annots_data["cams"]["D"]).astype(np.float32)
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

    def load_meta_data(self, frame_ids=None):
        data = torch.load(os.path.join(self.root_dir, "pose_data.pt"))

        keys = [
            "lbs_weights",
            "rest_verts",
            "rest_joints",
            "rest_tfs",
            "rest_tfs_bone",
            "verts",
            "joints",
            "tfs",
            "tf_bones",
            "params",
            "faces",
        ]
        return {
            key: (
                data[key][frame_ids].numpy()
                if (
                    frame_ids is not None
                    and key in ["verts", "joints", "tfs", "tf_bones", "params"]
                )
                else data[key].numpy()
            )
            for key in keys
        }

    def _process_mask(self, mask, border: int = 5, ignore_value: int = 128):
        kernel = np.ones((border, border), np.uint8)
        mask_erode = cv2.erode(mask.copy(), kernel)
        mask_dilate = cv2.dilate(mask.copy(), kernel)
        mask[mask_dilate != mask_erode] = ignore_value
        return mask

    def _create_splits(self):
        from utils.clustering import train_val_test_split

        meta_data = self.load_meta_data()
        
        # root transform from canonical to world. [N, 4, 4]
        transform_global = torch.from_numpy(
            meta_data["tf_bones"][:, 0] @ np.linalg.inv(meta_data["rest_tfs_bone"][0])
        ).float()
        verts = torch.from_numpy(meta_data["verts"]).float()
        
        # explicitly write out the random seed we used to create the split. The seed is 
        # generated from `verts.numel()`.
        seed = {
            313: 30384900, 315: 45163950, 377: 12753390, 386: 13352820,
            387: 13518180, 390: 24204570, 392: 11492520, 393: 13600860, 394: 17755530
        }[self.subject_id]
        print ("Creating data splits using seed %d. Will Save to %s" % (seed, self.splits_dir))

        splits = train_val_test_split(transform_global, verts, ncluster=10, seed=seed)        
        for split_name, split_ids in splits.items():
            fids = [self.frame_ids[i] for i in split_ids]
            os.makedirs(self.splits_dir, exist_ok=True)
            with open(
                os.path.join(self.splits_dir, "%s.txt" % split_name), "w"
            ) as fp:
                for fid in fids:
                    fp.write("%d\n" % fid)

if __name__ == "__main__":
    import nvdiffrast.torch as dr

    cfg_path = "./tmp/configs/zju_mocap_313.json"
    cfg = json.load(open(cfg_path, "r"))
    FLAGS = argparse.Namespace()
    FLAGS.spp = 1
    for k, v in cfg.items():
        setattr(FLAGS, k, v)
    FLAGS.resize_factor = 0.5
    FLAGS.pre_load = False
    FLAGS.local_rank = 0
    FLAGS.dev_run = True
    FLAGS.dev_run_num_frames = 10
    FLAGS.tfs_type = 'joint'
    # FLAGS.color_bkgd_aug = 'gray'
    FLAGS.subject_id = 377
    # FLAGS.dilate_erode_mask = False

    smpl_path = os.path.join("data", "SMPL_NEUTRAL.pkl")
    output_dir = "./tmp/imgs/zju_mocap"
    os.makedirs(output_dir, exist_ok=True)

    glctx = dr.RasterizeGLContext()
    for split in ['dev_run']:
        dataset = DatasetZJUMocap(FLAGS=FLAGS, split=split)

        animation_meta_data = dataset.get_animation_meta_data()
        for k, v in animation_meta_data.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}\t{v.dtype}\t{v.shape}")
        
        sample = dataset[0]
        batch = dataset.collate([dataset[i] for i in range(0, 46)])
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"{k} {v.dtype}, {v.shape}")
            else:
                print(f"{k} {type(v)} {v}")
        
        resolution = batch["resolution"]
        tfs_in_canon = batch["tfs_in_canon"]
        mvp = batch["mvp"]
        gt_img = batch["img"]
        verts = animation_meta_data["rest_verts_in_canon"]
        faces = animation_meta_data["faces"].int()
        lbs_weights = animation_meta_data["lbs_weights"]

        verts = torch.nn.functional.pad(verts, (0, 1), value=1.0)
        batch_size = batch["mv"].shape[0]
        verts = verts.unsqueeze(0).expand(batch_size, -1, -1)

        # Skinning
        T = torch.einsum("ij,bjkl->bikl", lbs_weights, tfs_in_canon) 
        verts = torch.einsum("bij,bikj->bik", verts, T)
        # Modeling, viewing, projection
        verts = torch.matmul(verts, mvp.transpose(1, 2))

        color = torch.randn_like(verts) * 0.2 + 0.8

        verts = verts.cuda()
        color = color.cuda()
        faces = faces.cuda()

        rast, _ = dr.rasterize(glctx, verts, faces, resolution)
        out, _ = dr.interpolate(color, rast, faces)

        from tqdm import trange
        from render.util import rgb_to_srgb
        for i in trange(len(out)):
            img = out.cpu().numpy()[i, :, :, :3]
            img = np.clip(img, 0.0, 1.0)
            _gt_img = rgb_to_srgb(gt_img).cpu().numpy()[i, :, :, :3]
            alpha = gt_img[i, :, :, 3].cpu().numpy()
            _gt_img = np.clip(_gt_img, 0.0, 1.0)
            _gt_img[alpha > 0] *= 1.5
             
            blend_img = img * 0.5 + _gt_img
            blend_img = np.clip(np.rint(blend_img * 255), 0, 255).astype(np.uint8)
            img = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8)
            _gt_img = np.clip(np.rint(_gt_img * 255), 0, 255).astype(np.uint8)
            blend_img = np.concatenate([img, _gt_img, blend_img], axis=1)
            imageio.imwrite(osp.join(output_dir, f'blend_test_zju_mocap_cam_split_{split}_cmr_{batch["camera_id"][i]}_frm_{batch["frame_id"][i]}.png'), blend_img)






