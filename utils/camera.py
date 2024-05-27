# Copyright (c) Meta Platforms, Inc. and affiliates.
import math
from typing import Tuple

import torch

from .structures import Cameras


def transform_cameras(cameras: Cameras, resize_factor: float) -> Cameras:
    intrins = cameras.intrins
    intrins[..., :2, :] = intrins[..., :2, :] * resize_factor
    width = int(cameras.width * resize_factor + 0.5)
    height = int(cameras.height * resize_factor + 0.5)
    return Cameras(
        intrins=intrins,
        extrins=cameras.extrins,
        distorts=cameras.distorts,
        width=width,
        height=height,
    )

