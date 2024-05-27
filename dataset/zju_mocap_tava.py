import os
import torch
import numpy as np
import logging
import math

from .dataset_utils import MetaData, check_to_log, get_transforms_t_pose2vitruvian_pose

from .zju_mocap import DatasetZJUMoCap, SubjectParser


LOGGER = logging.getLogger(__name__)


class DatasetZJUMoCapTAVA(DatasetZJUMoCap):
    VALID_SPLIT_NAMES = ['all', 'train', 'val_ind', 'val_ood', 'val_view', 'test', 'dev_run']    

    def __init__(
        self,
        *,
        root_dir,
        split,
        # mask_dir_name,
        # tfs_type,
        views=None,
        begin_frame_id=0,
        num_frames=None,
        frame_interval=None,
        # dilate_erode_mask=False,
        order_camera_first=True,
        # resize_factor=1.0,
        # opencv_camera=True,
        pre_load=False,
        force_reload=False,
        # no_image_inputs=False,
        # cam_near_far=[0.1, 1000.0],
        # width=1024,
        # height=1024,
        # use_vitruvian_pose=False,
        **kwargs,
    ):
        super().__init__(
            root_dir=root_dir,
            split=split,
            views=views,
            begin_frame_id=begin_frame_id,
            num_frames=num_frames,
            frame_interval=frame_interval,
            order_camera_first=order_camera_first,
            force_reload=force_reload,
            pre_load=False,
            **kwargs, 
        )

        if split not in self.VALID_SPLIT_NAMES:
            raise ValueError(f"Invalid split name: {split}, must in {self.VALID_SPLIT_NAMES}")

        if check_to_log():
            LOGGER.warning(f"Two args are not used: begin_frame_id: {begin_frame_id}, frame_interval: {frame_interval}")

        self.index_list = self.dataset_index_list_tava(
            num_frames, 
            split,
            views, 
            order_camera_first,
        )

        if pre_load:
            self._pre_load_data(root_dir, split, force_reload, "tava")

    def dataset_index_list_tava(self, num_frames, split, views=None, order_camera_first=True):
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

        if views is None:
            camera_ids = self._dataset_view_split(self.parser, split)
        else:
            camera_ids = views
        print(f"camera views: ({len(camera_ids)}): {camera_ids}")
        print(f"\tcamera e.g., {self.parser.image_files[0][camera_ids]}")

        print(f"order_camera_first: {order_camera_first}")
        frame_list = self._dataset_frame_split(split)

        index_list = []
        if order_camera_first:
            for frame_id in frame_list:
                index_list.extend([(frame_id, camera_id) for camera_id in camera_ids])
        else:
            for camera_id in camera_ids:
                index_list.extend([(frame_id, camera_id) for frame_id in frame_list])

        if check_to_log():
            LOGGER.warning(f"TAVA takes interval in (frames X cameras), not (frames) in Neural Body.")

        num_frames_cameras = len(index_list)
        if num_frames is None:
            num_frames = len(index_list)
            print(f"num_frames is None, set to {num_frames}.")

        frame_interval = math.ceil(num_frames_cameras / num_frames)
        print(f"frame_interval is None, set to ceil({num_frames_cameras} / {num_frames}) = {frame_interval}.")

        if frame_interval < 1:
            raise ValueError("frame_interval must be >= 1")

        index_list = index_list[::frame_interval]
        print(f"index_list ({len(index_list)}): {index_list[:5]}...{index_list[-5:]}")
        return index_list

    def _dataset_view_split(self, parser: SubjectParser, split: str):
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

    def _dataset_frame_split(self, split):
        if split in ["train", "val_view"]:
            splits_fp = os.path.join(self.parser.root_dir, "splits/train.txt")
        else:
            splits_fp = os.path.join(self.parser.root_dir, f"splits/{split}.txt")
        with open(splits_fp, mode="r") as fp:
            frame_list = np.loadtxt(fp, dtype=int).tolist()
        return frame_list


class SubjectParserTAVA(SubjectParser):
    def __init__(
            self,
            root_dir,
            mask_dir_name,
        ):
        super().__init__(root_dir, mask_dir_name)

        self.splits_dir = os.path.join(self.root_dir, "splits")
        if not os.path.exists(self.splits_dir):
            self._create_splits()

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