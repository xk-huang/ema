# Copyright (c) Meta Platforms, Inc. and affiliates.
import logging
import os

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity

LOGGER = logging.getLogger(__name__)
print = LOGGER.info


def save_ckpt(ckpt_dir, step, trainer, optimizer, optimizer_mesh, optimizer_motion, best=False):
    """Save the checkpoint."""
    if best:
        ckpt_path = os.path.join(ckpt_dir, "step-best.ckpt")  # in alphabetical order, number is always smaller than string
    else:
        ckpt_path = os.path.join(ckpt_dir, "step-%09d.ckpt" % step)

    if hasattr(trainer, "module"):
        trainer = trainer.module
    torch.save(
        {
            "step": step,
            "trainer": trainer.state_dict(),
            "optimizer": optimizer.state_dict(),
            "optimizer_mesh": optimizer_mesh.state_dict(),
            "optimizer_motion": optimizer_motion.state_dict(),
        },
        ckpt_path,
    )
    LOGGER.info("Save checkpoint to: %s" % ckpt_path)


def clean_up_ckpt(path, n_keep=5):
    """Clean up the checkpoints to keep only the last few (also keep the best one)."""
    if not os.path.exists(path):
        return
    ckpt_paths = sorted(
        [os.path.join(path, fp) for fp in os.listdir(path) if ".ckpt" in fp]
    )
    if len(ckpt_paths) > n_keep:
        for ckpt_path in ckpt_paths[:-n_keep]:
            LOGGER.warning("Remove checkpoint: %s" % ckpt_path)
            os.remove(ckpt_path)

def resume_from_ckpt(path, trainer, optimizer=None, optimizer_mesh=None, optimizer_motion=None, step=None, strict=False, load_geometry=True, load_light=True, load_material=True, load_geometry_non_rigid_offset_net=True):
    """Resume the model & optimizer from the latest/specific checkpoint.

    Return:
        the step of the ckpt. return 0 if no ckpt found.
    """
    if not os.path.exists(path):
        return 0
    if step is not None:
        ckpt_paths = [os.path.join(path, "step-%09d.ckpt" % step)]
        assert os.path.exists(ckpt_paths[0])
    else:
        ckpt_paths = sorted(
            [os.path.join(path, fp) for fp in os.listdir(path) if ".ckpt" in fp]
        )
    if len(ckpt_paths) > 0:
        ckpt_path = ckpt_paths[-1]
        ckpt_data = torch.load(ckpt_path, map_location="cpu")

        # align with the params shape in ckpt
        def getattr_by_dot(obj, name_chain, defualt=None):
            for name in name_chain.split("."):
                obj = getattr(obj, name, defualt)
            return obj

        reinit_optim = False
        for key in trainer.state_dict():
            if (key in ckpt_data["trainer"].keys() or f"module.{key}" in ckpt_data["trainer"].keys()) and trainer.state_dict()[key].shape != ckpt_data["trainer"][key].shape:
                print(f"Shape mismatch for {key}: {trainer.state_dict()[key].shape} vs {ckpt_data['trainer'][key].shape}")

                param = getattr_by_dot(trainer, key)
                param.data = ckpt_data['trainer'][key].data.to(param.device)

                reinit_optim = True
        if reinit_optim:
            raise RuntimeWarning(f"Reinitializing optimizer due to shape mismatch, not loading full weights.")

        # trainer.load_state_dict(
        #     {
        #         key.replace("module.", ""): value
        #         for key, value in ckpt_data["trainer"].items()
        #     },
        #     strict=strict
        # )
        if trainer.material is not None and load_material is True:
            print("Loading material")
            trainer.material.load_state_dict({k.replace("module.", "").replace("material.", ""): v for k, v in ckpt_data["trainer"].items() if k.startswith('material')})
        else:
            print("Not loading geometry")
        if trainer.geometry is not None and load_geometry is True:
            print("Loading geometry")
            trainer.geometry.load_state_dict({k.replace("module.", "").replace("geometry.", ""): v for k, v in ckpt_data["trainer"].items() if k.startswith('geometry')}, strict=strict)
        else:
            print("Not loading geometry")
        if trainer.geometry.non_rigid_offset_net is not None and load_geometry_non_rigid_offset_net is True:
            print("Loading geometry non_rigid_offset_net")
            trainer.geometry.non_rigid_offset_net.load_state_dict({k.replace("module.", "").replace("geometry.non_rigid_offset_net.", ""): v for k, v in ckpt_data["trainer"].items() if k.startswith('geometry.non_rigid_offset_net')})
        else:
            print("Not loading geometry non_rigid_offset_net")
        if trainer.light is not None and load_light is True:
            print("Loading light")
            trainer.light.load_state_dict({k.replace("module.", "").replace("light.", ""): v for k, v in ckpt_data["trainer"].items() if k.startswith('light')})
        else:
            print("Not loading light")

        if optimizer is not None:
            optimizer.load_state_dict(ckpt_data["optimizer"])
        if optimizer_mesh is not None:
            optimizer_mesh.load_state_dict(ckpt_data["optimizer_mesh"])
        if optimizer_motion is not None:
            optimizer_motion.load_state_dict(ckpt_data["optimizer_motion"])
        step = ckpt_data["step"]
        LOGGER.info("Load model from checkpoint: %s" % ckpt_path)
    else:
        step = -1
        LOGGER.info(f"fail to load at {path}")
    return step
