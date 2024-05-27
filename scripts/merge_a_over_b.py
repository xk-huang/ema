import argparse
import imageio
from glob import glob
import os
import os.path as osp
from tqdm import tqdm


new_img_dir = "outputs/demo-efficiency-h36m/texture/9/230315_130515/validate_novel_view/opt"
new_mask_dir = "outputs/demo-efficiency-h36m/texture/9/230315_130515/validate_novel_view/opt_alpha"

old_img_dir = "outputs/demo-efficiency-h36m/texture-origin/9/230315_131032/dmtet_validate_novel_view/opt"

out_img_dir = "outputs/demo-efficiency-h36m/texture-edited/"

new_img_paths = sorted(glob(osp.join(new_img_dir, "*.png")))
new_mask_paths = sorted(glob(osp.join(new_mask_dir, "*.png")))
old_img_paths = sorted(glob(osp.join(old_img_dir, "*.png")))

os.makedirs(out_img_dir, exist_ok=True)

for new_img_path, new_mask_path, old_img_path in tqdm(zip(new_img_paths,new_mask_paths, old_img_paths)):
    new_img = imageio.imread(new_img_path)
    new_mask = imageio.imread(new_mask_path)
    old_img = imageio.imread(old_img_path)
    old_img[new_mask>0] = new_img[new_mask>0]
    out_img_path = osp.join(out_img_dir, osp.basename(new_img_path))
    imageio.imwrite(out_img_path, old_img)