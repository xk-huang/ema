import logging
import math
import os
import os.path as osp

import hydra
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pytorch3d
import torch
from omegaconf import OmegaConf
import cv2

# Util function for loading meshes
from pytorch3d.io import load_obj, load_objs_as_meshes
from pytorch3d.renderer import (
    DirectionalLights,
    FoVPerspectiveCameras,
    Materials,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,
    look_at_view_transform,
)

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.utils import cameras_from_opencv_projection
from tqdm import tqdm

from render.mesh import Mesh, load_mesh
from render.util import rgb_to_srgb, srgb_to_rgb
from scripts.utils.lbs import batch_rodrigues, batch_rigid_transform

try:
    from .dataset_utils import (
        MetaData,
        check_to_log,
        get_transforms_t_pose2vitruvian_pose,
    )
except ImportError:
    from dataset_utils import (
        MetaData,
        check_to_log,
        get_transforms_t_pose2vitruvian_pose,
    )
try:
    from .zju_mocap import DatasetZJUMoCap, SubjectParser
except ImportError:
    from dataset.zju_mocap import DatasetZJUMoCap, SubjectParser
try:
    from .constants import SMPL_CONSTANTS
except ImportError:
    from dataset.constants import SMPL_CONSTANTS

LOGGER = logging.getLogger(__name__)


class DatasetZJUMoCapSynthesis(torch.utils.data.Dataset):
    def __init__(
        self,
        texture_obj_path,
        force_generate_synthesis_data=False,
        add_noise=False,
        noise_scale=0.01,
        **kwargs,
    ):
        if "_target_" in kwargs:
            del kwargs["_target_"]
        self.zju_mocap_dataset = DatasetZJUMoCap(**kwargs)
        self.texture_obj_path = texture_obj_path
        LOGGER.info(f"Texture obj path: {self.texture_obj_path}")

        self.split = self.zju_mocap_dataset.split
        self.synthesis_rgba_dir = osp.join(
            self.zju_mocap_dataset.parser.root_dir, "synthesis_rgba", self.split
        )
        if force_generate_synthesis_data or not osp.exists(self.synthesis_rgba_dir):
            self.generate_synthesis_data()

        self.no_image_inputs = False
        if self.split == "train" and add_noise:
            print("Adding noise to training data")
            self.add_noise_to_tfs_params(noise_scale)

    def add_noise_to_tfs_params(self, noise_scale):
        print(f"noise_scale: {noise_scale}")
        rest_verts = self.zju_mocap_dataset.meta_data["rest_verts"][None]
        rest_joints = self.zju_mocap_dataset.meta_data["rest_joints"][None]
        noise_idx_ls = []
        frame_ids = set(
            [frame_id for (frame_id, _) in self.zju_mocap_dataset.index_list]
        )
        for frame_id in frame_ids:
            if torch.randint(0, 10, ()) < 1:
                continue
            params = self.zju_mocap_dataset.meta_data["params"][
                frame_id : frame_id + 1
            ].clone()
            pose, Rh, Th = params[:, :72], params[:, 72:75], params[:, 75:78]
            target_joint_idx = torch.randint(1, 24, ())
            noise = (torch.rand(()) * 2 - 1) * noise_scale
            pose[:, target_joint_idx * 3 : target_joint_idx * 3 + 3] += noise

            rot_mats = batch_rodrigues(pose.view(-1, 24, 3))
            parants = torch.tensor([-1] + SMPL_CONSTANTS.BONE_HEAD_JOINT_IDS)
            _, tfs, _ = batch_rigid_transform(rot_mats, rest_joints, parants)

            rot = batch_rodrigues(Rh)
            transl = Th.unsqueeze(dim=1)
            global_transform = torch.eye(4, dtype=rot.dtype, device=rot.device)
            global_transform[:3, :3] = rot * 1.0
            global_transform[:3, 3] = transl
            tfs = torch.einsum("ij,...jk->...ik", global_transform, tfs)

            self.zju_mocap_dataset.meta_data["params"][
                frame_id : frame_id + 1, :72
            ] = pose
            self.zju_mocap_dataset.meta_data["tfs"][frame_id : frame_id + 1] = tfs
            noise_idx_ls.append(frame_id)
        print(
            f"Added noise to {len(noise_idx_ls)} ([{noise_idx_ls[:3]} ... {noise_idx_ls[-3:]}) frames out of {len(frame_ids)} frames"
        )

    def generate_synthesis_data(self):
        LOGGER.info(
            f"Generating synthesis data for {self.split} split at {self.synthesis_rgba_dir}"
        )
        os.makedirs(self.synthesis_rgba_dir, exist_ok=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        mesh = load_objs_as_meshes([self.texture_obj_path], device=device)

        height = self.zju_mocap_dataset.height
        width = self.zju_mocap_dataset.width

        raster_settings = RasterizationSettings(
            image_size=(height, width),
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=None, raster_settings=raster_settings),
            shader=SoftPhongShader(device=device, cameras=None, lights=lights),
        )

        image_size = torch.tensor([[height, width]]).to(device)

        with torch.no_grad():
            for idx, data in enumerate(tqdm(self.zju_mocap_dataset)):
                camera_id = data["camera_id"]
                frame_id = data["frame_id"]
                verts_in_model = data["verts_in_model"].to(device)

                K = self.zju_mocap_dataset.parser.cameras[camera_id]["K"].copy()
                # K[:2, :] = K[:2, :] * self.zju_mocap_dataset.resize_factor
                w2c = self.zju_mocap_dataset.parser.cameras[camera_id]["w2c"].copy()
                R = w2c[:3, :3]
                T = w2c[:3, 3]

                R, T, K = (
                    torch.from_numpy(R)[None].to(device),
                    torch.from_numpy(T)[None].to(device),
                    torch.from_numpy(K)[None].to(device),
                )

                cameras = pytorch3d.utils.cameras_from_opencv_projection(
                    R=R, tvec=T, camera_matrix=K, image_size=image_size
                ).to(device)

                posed_mesh = mesh.update_padded(verts_in_model)
                images = renderer(posed_mesh, cameras=cameras)
                images = images[0].cpu().numpy()
                images[..., -1] = images[..., -1] > 0.0
                images = (images * 255).astype(np.uint8)
                imageio.imwrite(
                    osp.join(self.synthesis_rgba_dir, f"{camera_id}-{frame_id}.png"),
                    images,
                )

    def generate_dummy_data(self):
        device = "cpu"
        data = self.zju_mocap_dataset[
            torch.randint(0, len(self.zju_mocap_dataset), ()).item()
        ]
        mesh = load_objs_as_meshes([self.texture_obj_path], device=device)
        # cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

        camera_id = data["camera_id"]
        K = self.zju_mocap_dataset.parser.cameras[camera_id]["K"].copy()
        w2c = self.zju_mocap_dataset.parser.cameras[camera_id]["w2c"].copy()
        R = w2c[:3, :3]
        T = w2c[:3, 3]
        image_size = torch.tensor(
            [self.zju_mocap_dataset.height, self.zju_mocap_dataset.width]
        )

        R, T, K, image_size = (
            torch.from_numpy(R)[None],
            torch.from_numpy(T)[None],
            torch.from_numpy(K)[None],
            image_size[None],
        )

        # D = self.zju_mocap_dataset.parser.cameras[camera_id]["D"].copy()
        cameras = pytorch3d.utils.cameras_from_opencv_projection(
            R=R, tvec=T, camera_matrix=K, image_size=image_size
        )

        raster_settings = RasterizationSettings(
            image_size=512,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=SoftPhongShader(device=device, cameras=cameras, lights=lights),
        )
        new_mesh = mesh.update_padded(data["verts_in_model"])
        images = renderer(new_mesh)
        f, ax = plt.subplots(1, 2)
        ax[0].imshow(images[0, ..., :3].cpu().numpy())
        ax[0].axis("off")
        ax[1].imshow(rgb_to_srgb(data["img"][0, ..., :3]).cpu().numpy())
        ax[1].axis("off")
        plt.savefig(
            "tmp/syn_zju.png",
            bbox_inches="tight",
        )

    def __len__(self):
        return len(self.zju_mocap_dataset)

    def __getitem__(self, index):
        data = self.zju_mocap_dataset[index]
        frame_id, camera_id = data["frame_id"], data["camera_id"]
        rgba_path = osp.join(self.synthesis_rgba_dir, f"{camera_id}-{frame_id}.png")

        rgba = imageio.imread(rgba_path)
        rgba = cv2.resize(
            rgba,
            (0, 0),
            fx=self.zju_mocap_dataset.resize_factor,
            fy=self.zju_mocap_dataset.resize_factor,
            interpolation=cv2.INTER_AREA,
        )
        rgba = torch.from_numpy(rgba).to(self.zju_mocap_dataset.dtype) / 255.0
        rgba[..., :3] = srgb_to_rgb(rgba[..., :3])

        data["real_img"] = data["img"]
        data["img"] = rgba[None]

        return data

    EXCEPT_KEYS = ["subject_id", "camera_id", "frame_id", "resolution", "spp", "img"]

    def collate(self, batch):
        collate_batch = {
            "resolution": batch[0]["resolution"],
            # 'spp': batch[0]['spp'],
            "img": torch.cat(list([item["img"] for item in batch]), dim=0)
            if not self.no_image_inputs
            else None,
            "camera_id": [item["camera_id"] for item in batch],  # List[int]
            "frame_id": [item["frame_id"] for item in batch],  # List[int]
        }

        for k in batch[0].keys():
            if k in self.EXCEPT_KEYS:
                continue
            collate_batch[k] = torch.cat([item[k] for item in batch], dim=0)

        return collate_batch

    def get_animation_meta_data(self):
        return self.zju_mocap_dataset.get_animation_meta_data()


if __name__ == "__main__":
    # @hydra.main(config_path=CONF_FP, config_name="base", version_base="1.2")
    # CONF_FP: str = os.path.join(os.path.dirname(__file__), "configs")
    hydra.initialize(
        config_path="../configs", version_base="1.2", job_name="test_synthesis"
    )
    FLAGS = hydra.compose(
        config_name="base",
        overrides=[
            "dataset@_global_=zju_mocap/313",
            "+train_dataset.texture_obj_path=data/toy_data_zju_format/SMPL/SMPL_female_default_resolution.obj",
            "+train_dataset.add_noise=true",
            "+train_dataset.noise_scale=1",
            "+train_dataset.force_generate_synthesis_data=false",
            "train_dataset._target_=dataset.zju_mocap_synthesis.DatasetZJUMoCapSynthesis",
            "+validate_dataset.0.texture_obj_path='data/toy_data_zju_format/SMPL/SMPL_female_default_resolution.obj'",
            "validate_dataset.0._target_=dataset.zju_mocap_synthesis.DatasetZJUMoCapSynthesis",
            "+validate_dataset.0.force_generate_synthesis_data=false",
            "+validate_dataset.1.texture_obj_path='data/toy_data_zju_format/SMPL/SMPL_female_default_resolution.obj'",
            "validate_dataset.1._target_=dataset.zju_mocap_synthesis.DatasetZJUMoCapSynthesis",
            "+validate_dataset.1.force_generate_synthesis_data=false",
        ],
    )
    print(OmegaConf.to_yaml(FLAGS))

    # dataset_train = hydra.utils.instantiate(FLAGS.train_dataset)
    # dataset_validate = [hydra.utils.instantiate(cfg) for cfg in FLAGS.validate_dataset]
    dataset_train = DatasetZJUMoCapSynthesis(**FLAGS.train_dataset)
    dataset_validate = [
        DatasetZJUMoCapSynthesis(**cfg) for cfg in FLAGS.validate_dataset
    ]

    print(f"len(dataset_train): {len(dataset_train)}")
    # exit()

    data = dataset_train[1]

    device = "cuda"
    camera_id = data["camera_id"]
    frame_id = data["frame_id"]
    verts_in_model = data["verts_in_model"].to(device)

    mesh = load_objs_as_meshes([dataset_train.texture_obj_path], device=device)
    image_size = (1024, 1024)
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=None, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=None, lights=lights),
    )

    K = dataset_train.zju_mocap_dataset.parser.cameras[camera_id]["K"].copy()
    # K[:2, :] = K[:2, :] * self.zju_mocap_dataset.resize_factor
    w2c = dataset_train.zju_mocap_dataset.parser.cameras[camera_id]["w2c"].copy()
    R = w2c[:3, :3]
    T = w2c[:3, 3]

    R, T, K = (
        torch.from_numpy(R)[None].to(device),
        torch.from_numpy(T)[None].to(device),
        torch.from_numpy(K)[None].to(device),
    )

    cameras = pytorch3d.utils.cameras_from_opencv_projection(
        R=R, tvec=T, camera_matrix=K, image_size=torch.tensor(image_size)[None]
    ).to(device)

    def post_process(images):
        images = images[0].cpu().numpy()
        images[..., -1] = images[..., -1] > 0.0
        images = (images * 255).astype(np.uint8)
        return images

    original_image = imageio.imread(
        osp.join(dataset_train.synthesis_rgba_dir, f"{camera_id}-{frame_id}.png")
    )

    mesh_verts_in_model = mesh.update_padded(verts_in_model)
    image_mesh_verts_in_model = renderer(mesh_verts_in_model, cameras=cameras)
    image_mesh_verts_in_model = post_process(image_mesh_verts_in_model)
    imageio.imwrite("tmp/053123.origin_smpl.png", image_mesh_verts_in_model)

    noise_images = []
    for noise in [1e-2 * i for i in range(0, 3)]:
        tfs = data["tfs_in_canon"]
        params = data["params"]
        pose, Rh, Th = params[:, :72].clone(), params[:, 72:75], params[:, 75:78]
        pose[:, 6 * 3 : 6 * 3 + 3] += noise

        rest_verts = dataset_train.zju_mocap_dataset.meta_data["rest_verts"]
        rest_joints = dataset_train.zju_mocap_dataset.meta_data["rest_joints"]
        lbs_weights = dataset_train.zju_mocap_dataset.meta_data["lbs_weights"]
        rest_joints = rest_joints[None]
        rest_verts = rest_verts[None]
        batch_size = 1
        num_joints = 24

        rot_mats = batch_rodrigues(pose.view(-1, 24, 3))
        parants = torch.tensor([-1] + SMPL_CONSTANTS.BONE_HEAD_JOINT_IDS)
        _, tfs_, _ = batch_rigid_transform(rot_mats, rest_joints, parants)
        W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
        T = torch.matmul(W, tfs_.view(batch_size, num_joints, 16)).view(
            batch_size, -1, 4, 4
        )
        rot = batch_rodrigues(Rh)
        transl = Th.unsqueeze(dim=1)
        homogen_coord = torch.ones(
            [batch_size, rest_verts.shape[1], 1],
            dtype=rest_verts.dtype,
            device=rest_verts.device,
        )
        v_posed_homo = torch.cat([rest_verts, homogen_coord], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))
        verts = v_homo[:, :, :3, 0]
        verts = torch.matmul(verts, rot.transpose(1, 2)) + transl

        mesh_verts = mesh.update_padded(verts.to(device))
        image_verts = renderer(mesh_verts, cameras=cameras)
        image_verts = post_process(image_verts)
        noise_images.append(image_verts)
    imageio.imwrite(
        "tmp/053123.compare_smpl.png",
        np.concatenate(
            [original_image, noise_images[0]]
            + [original_image / 2 + noise_image / 2 for noise_image in noise_images],
            axis=-2,
        ),
    )
