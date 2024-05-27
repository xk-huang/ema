import os

import cv2
import distinctipy
import hydra
import imageio
import nvdiffrast.torch as dr
import torch
import tqdm
from flask import Flask, Response
from omegaconf import DictConfig, ListConfig

from render import util

app = Flask(__name__)

@app.route('/')
def display():
    return Response(_display(), mimetype='multipart/x-mixed-replace; boundary=frame')

def _display(no_flask=False):
    global FLAGS
    if isinstance(FLAGS.validate_dataset, DictConfig):
        FLAGS.validate_dataset = ListConfig([FLAGS.validate_dataset])
    splits = [dataset.split for dataset in FLAGS.validate_dataset]
    dataset_validate = [hydra.utils.instantiate(cfg) for cfg in FLAGS.validate_dataset]
    re_forward_smpl = FLAGS.get("re_forward_smpl", False)
    no_overlap_smpl = FLAGS.get("no_overlap_smpl", False)
    no_smpl = FLAGS.get("no_smpl", False)
    print(f"split: {splits}")

    glctx = dr.RasterizeGLContext()
    
    for dataset in dataset_validate:
        hydra.utils.instantiate(FLAGS.train_dataset)
        print(f"Split: {dataset.split}, num images: {len(dataset)}")

        animation_meta_data = dataset.get_animation_meta_data()
        verts = animation_meta_data["rest_verts_in_canon"]
        faces = animation_meta_data["faces"].int()
        lbs_weights = animation_meta_data["lbs_weights"]

        skinning_composite_factor = 0.5
        color = torch.ones(verts.shape[0], 4, dtype=torch.float32) * skinning_composite_factor  # Must be (..., 4) to rasterize mask
        color[..., :3] = skin_weights2color(lbs_weights)

        verts_cpu = torch.nn.functional.pad(verts, (0,1), value=1.0)
        verts, faces, lbs_weights, color = verts.cuda(), faces.cuda(), lbs_weights.cuda(), color.cuda()
        
        for data in tqdm.tqdm(dataset):
            # [NOTE] Not the same, we do not use beta
            # my_verts_in_model = torch.matmul(verts_cpu, data["global_model_transformation"].transpose(1, 2))
            # verts_in_model = data["verts_in_model"]
            # tqdm.tqdm.write(f"diff: {torch.mean(torch.abs(my_verts_in_model[..., :3] - verts_in_model)).item()}")

            rgba = data["img"][0]
            rgba[..., :3] = util.rgb_to_srgb(rgba[..., :3])
            rgb, alpha = rgba[..., :3].clone(), rgba[..., 3].clone()
            rgb[alpha == 0] =  0.

            # SMPL
            if not no_smpl:
                smpl_rgba = render_smpl(glctx, data, verts, faces, lbs_weights, color, re_forward_smpl).cpu()
                smpl_rgb, smpl_alpha = smpl_rgba[..., :3], smpl_rgba[..., 3:]
                if not no_overlap_smpl:
                    rgb = rgb * (1 - smpl_alpha) + smpl_rgb * smpl_alpha
                rgb = torch.cat([rgb, smpl_rgb], axis=-2)


            rgb_numpy = (rgb * 255).numpy().astype("uint8")
            rgb_numpy = cv2.cvtColor(rgb_numpy, cv2.COLOR_RGB2BGR)

            if no_flask:
                yield rgb_numpy
            else:
                _, frame = cv2.imencode('.jpg', rgb_numpy)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n')

        print("Done")
        del dataset

def render_smpl(glctx, batch, verts, faces, lbs_weights, color, re_forward_smpl=False):
        resolution = batch["resolution"]
        tfs_in_canon = batch["tfs_in_canon"].cuda()
        mvp = batch["mvp"].cuda()
        vp = batch["vp"].cuda()

        batch_size = mvp.shape[0]

        if re_forward_smpl is True:
            verts = torch.nn.functional.pad(verts, (0, 1), value=1.0)
            verts = verts.unsqueeze(0).expand(batch_size, -1, -1)
            # Skinning
            T = torch.einsum("ij,bjkl->bikl", lbs_weights, tfs_in_canon) 
            verts = torch.einsum("bij,bikj->bik", verts, T)

            # Modeling, viewing, projection
            verts = torch.matmul(verts, mvp.transpose(1, 2))
        else:
            verts = batch["verts_in_model"].cuda()
            verts = torch.nn.functional.pad(verts, (0, 1), value=1.0)
            verts = torch.matmul(verts, vp.transpose(1, 2))
        
        rast, _ = dr.rasterize(glctx, verts, faces, resolution)
        out, _ = dr.interpolate(color, rast, faces)

        return out[0]

def skin_weights2color(weights):
    num_colors = weights.shape[1]
    color_idx = weights.max(dim=-1)[1]

    colors = distinctipy.get_colors(num_colors, pastel_factor=0.5, rng=1)
    colors = torch.tensor(colors, dtype=torch.float32, device=weights.device)

    return colors[color_idx]

@hydra.main(config_path="../configs", config_name="base", version_base="1.2")
def main(_FLAGS):
    print("Additional parameters: no_flask: bool, video_dir: str, num_Frames, int.")
    global FLAGS
    FLAGS = _FLAGS
    if FLAGS.get("no_flask", False):
        print("No flask")

        video_dir = FLAGS.get("video_dir", "tmp/zju_smpl_vis")
        os.makedirs(video_dir, exist_ok=True)


        num_frames = FLAGS.get("num_frames", 300)
        
        if num_frames == 1:
            video_path = os.path.join(video_dir, f"{FLAGS.subject_id}{FLAGS.get('video_suffix', '')}.png")
            print(f"video_path: {video_path}")
            imageio.imwrite(video_path, next(_display(no_flask=True)))
        else:
            video_path = os.path.join(video_dir, f"{FLAGS.subject_id}{FLAGS.get('video_suffix', '')}.mp4")
            print(f"video_path: {video_path}")
            cnt = 0
            writer = imageio.get_writer(video_path, fps=50)
            for img in _display(no_flask=True):
                if cnt >= num_frames:
                    break
                writer.append_data(img)
                cnt += 1

            writer.close()
        return

    app.run(host='0.0.0.0', port=80, threaded=True)


if __name__ == "__main__":
    main()
