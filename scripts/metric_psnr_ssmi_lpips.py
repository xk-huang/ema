import argparse
import multiprocessing as mp
import os.path as osp
from glob import glob
from multiprocessing import Manager, Pool

import imageio
import lpips
import numpy as np
import skimage.metrics
import torch
from tqdm import tqdm


# [NOTE](xk): This scripts does not consider the crop of the image. Thus the metrics are higher than the real ones.

def job_psnr_ssim(var):
    """Job for PSNR and SSIM. Parrallel computing of PSNR & SSIM."""
    test, gt, psnr_ls, ssim_ls, test_all, gt_all = var
    test = imageio.imread(test)[..., :3]
    gt = imageio.imread(gt)[..., :3]
    psnr = skimage.metrics.peak_signal_noise_ratio(test, gt)
    ssim = skimage.metrics.structural_similarity(test, gt, multichannel=True)

    psnr_ls.append(psnr)
    ssim_ls.append(ssim)

    test_all.append(test)
    gt_all.append(gt)


if __name__ == "__main__":
    # Set up config
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", "-t", type=str)
    parser.add_argument("--gt_dir", "-g", type=str)
    parser.add_argument("--exp_name", type=str, default="exp")
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--ext", type=str, default=".png")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="image_size=512x512, batch_size=32 -> 21G.",
    )
    args = parser.parse_args()

    # Set up directory and image paths
    test_dir, gt_dir = args.test_dir, args.gt_dir
    test_ls = sorted(glob(osp.join(test_dir, f"*{args.ext}")))
    gt_ls = sorted(glob(osp.join(gt_dir, f"*{args.ext}")))
    exp_name = args.exp_name

    if len(test_ls) == 0 or len(gt_ls) == 0:
        raise ValueError(f"{len(test_ls)}, {len(gt_ls)}, \n{test_dir}, {gt_dir}")
    if len(test_ls) != len(gt_ls):
        min_len = min(len(test_ls), len(gt_ls))
        test_ls, gt_ls = test_ls[:min_len], gt_ls[:min_len]

    # PSNR & SSIM
    mp.set_start_method("spawn")
    with Manager() as shm:
        psnr_ls = shm.list()
        ssim_ls = shm.list()

        test_all = shm.list()
        gt_all = shm.list()

        with Pool(4) as p:
            p.map(
                job_psnr_ssim,
                tqdm(
                    [
                        (test, gt, psnr_ls, ssim_ls, test_all, gt_all)
                        for test, gt in zip(test_ls, gt_ls)
                    ]
                ),
            )

        psnr_avg = np.sum(psnr_ls) / len(psnr_ls)
        ssim_avg = np.sum(ssim_ls) / len(ssim_ls)
        test_all = tuple(test_all)
        gt_all = tuple(gt_all)

    # LPIPS
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_all = torch.stack(
        [torch.from_numpy(i).permute(2, 0, 1).to(dtype=torch.float32) for i in test_all]
    )
    gt_all = torch.stack(
        [torch.from_numpy(i).permute(2, 0, 1).to(dtype=torch.float32) for i in gt_all]
    )
    test_split = torch.split(test_all, args.batch_size, dim=0)
    gt_split = torch.split(gt_all, args.batch_size, dim=0)

    lpips_all = []
    lpips_vgg = lpips.LPIPS(net="vgg").to(device=device)
    with torch.no_grad():
        for predi, gti in zip(test_split, gt_split):
            lpips_i = lpips_vgg(predi.to(device=device), gti.to(device=device))
            lpips_all.append(lpips_i)
        lpips_ = torch.cat(lpips_all)
    lpips_ = lpips_.mean().item()

    # Logging
    print(f"[{exp_name}]\tpsnr: {psnr_avg:.2f}\tssim: {ssim_avg:.4f}\tlpips: {lpips_:.4f}\n")

    if args.log_file is not None:
        with open(args.log_file, "a") as f:
            f.write(
                f"[{exp_name}:{test_dir}]\tpsnr: {psnr_avg:.2f}\tssim: {ssim_avg:.4f}\tlpips: {lpips_:.4f}\n"
            )
