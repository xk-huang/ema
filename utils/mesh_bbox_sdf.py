import os
import trimesh
import numpy as np
import cv2
import sdf
import click
import logging

LOGGER = logging.getLogger(__name__)
print = LOGGER.info

def get_mesh_bbox_sdf(verts, faces, edge_process='dilate', render_scale=0.95, padding_percent=0.02, image_height=1000, aspect_ratio=1.0, nvdiffrast_context=None, debug=False, verbose=False):
    # get normalized mesh
    original_vertices_range = verts.ptp(axis=0)
    original_vertices_center = original_vertices_range / 2 + verts.min(axis=0)

    normalized_mesh, normalized_verts_height = normalize_mesh(verts, faces, render_scale)

    print(f"normalized_verts_height: {normalized_verts_height}")

    print(f"image_height: {image_height}")
    print(f"aspect_ratio: {aspect_ratio}")
    print(f"render_scale: {render_scale}")
    print(f"padding_percent: {padding_percent * 100} %")

    kernel_size = max(int(image_height * render_scale * padding_percent), 0)
    print(f"kernel_size: {kernel_size}")

    # get tri-view meshes
    meshes = []
    meshes.append(normalized_mesh)

    rotated_normalized_mesh = normalized_mesh.copy()
    angle = np.pi / 2
    direction = [0, 1, 0]
    rot_matrix = trimesh.transformations.rotation_matrix(angle, direction)
    rotated_normalized_mesh.apply_transform(rot_matrix)

    meshes.append(rotated_normalized_mesh)

    rotated_normalized_mesh = normalized_mesh.copy()
    angle = np.pi / 2
    direction = [1, 0, 0]
    rot_matrix = trimesh.transformations.rotation_matrix(angle, direction)
    rotated_normalized_mesh.apply_transform(rot_matrix)

    meshes.append(rotated_normalized_mesh)

    # get tri-view images
    triviews = []
    triviews_all = []
    original_over_enlarge_scales = []

    print("setup renderer")
    for mesh in meshes:
        if nvdiffrast_context is None:
            depth = render_mesh_by_pyrender(mesh, width=aspect_ratio * image_height, height=image_height)
        else:
            depth = render_mesh_by_nvdiffrast(nvdiffrast_context, mesh, width=aspect_ratio * image_height, height=image_height)

        if debug:
            from IPython.core.debugger import set_trace; set_trace()

        binary = (depth > 0).astype("uint8") * 255

        contour = cv2.findContours(binary.astype("uint8"), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contour = contour[0] if len(contour) == 2 else contour[1]
        contour = sorted(contour, key=cv2.contourArea)
        out_mask = np.zeros_like(binary.astype("uint8"))
        contour_binary = (cv2.drawContours(out_mask, [contour[-1]], -1, 255, cv2.FILLED, 1) > 0).astype("uint8") * 255

        # [FIXME] is smooth necessary?
        # guassian_kernal_size = 2
        # smoothed_binary = cv2.GaussianBlur(contour_binary, (0,0), sigmaX=guassian_kernal_size, borderType = cv2.BORDER_DEFAULT)
        # smoothed_binary = (smoothed_binary > 0).astype("uint8") * 255

        if  edge_process == None or edge_process == 'none' or kernel_size == 0:
            edge_processed_binary = contour_binary
            print("edge_process: none")
        elif edge_process == 'dilate':
            dilate_kernel = np.ones((kernel_size, kernel_size),np.uint8)
            edge_processed_binary = cv2.dilate(contour_binary, dilate_kernel)
            print(f"edge_process: {edge_process}")
        elif edge_process == 'erode':
            # [NOTE] eroded tri-plane mesh might not totaly inside the mesh
            LOGGER.warning("erode might not totaly inside the mesh")
            dilate_kernel = np.ones((kernel_size, kernel_size),np.uint8)
            edge_processed_binary = cv2.erode(contour_binary, dilate_kernel)
            print(f"edge_process: {edge_process}")
        else:
            raise ValueError(f"edge_process {edge_process} not supported")
        edge_processed_binary = (edge_processed_binary > 0).astype("uint8") * 255

        if verbose:
            new_img_1 = np.concatenate((binary, contour_binary), axis=1)
            if edge_process == 'erode':
                layered_img = contour_binary.copy()
                layered_img[layered_img > 0] = 255
                layered_img[edge_processed_binary > 0] = 125
            elif edge_process == 'dilate':
                layered_img = edge_processed_binary.copy()
                layered_img[layered_img > 0] = 125
                layered_img[contour_binary > 0] = 255
            else:
                layered_img = np.zeros_like(binary)
            new_img_2 = np.concatenate((edge_processed_binary, layered_img), axis=1)
            new_img = np.concatenate((new_img_1, new_img_2), axis=0)
            triviews_all.append(new_img)

        if edge_process == "dilate":
            edge_processed_binary = cv2.copyMakeBorder(edge_processed_binary, kernel_size, kernel_size, kernel_size, kernel_size, cv2.BORDER_CONSTANT, value=0)
        triviews.append(edge_processed_binary)
        original_over_enlarge_scale = (np.array(np.where(binary)).ptp(axis=-1) / np.array(np.where(edge_processed_binary)).ptp(axis=-1))
        print(f"original_over_enlarge_scale: {original_over_enlarge_scale}")
        original_over_enlarge_scales.append(original_over_enlarge_scale)


    # generate sdf
    f0= sdf.image(triviews[0]).extrude(1)
    f1 = sdf.image(triviews[1]).extrude(1).rotate(-np.pi / 2, (0, 1, 0))
    f2 = sdf.image(triviews[2]).extrude(1).rotate(-np.pi / 2, (1, 0, 0))

    f = sdf.intersection(f0, f1, f2)
    f_triangle_faces = f.generate(step=0.01)
    f_verts = np.unique(f_triangle_faces, axis=0)
    f_verts_bounds = f_verts.ptp(axis=0)
    f_verts_center = (f_verts_bounds / 2 + f_verts.min(axis=0))

    f = f.translate(-f_verts_center)
    f = f.scale(1 / f_verts_bounds[1] / original_over_enlarge_scales[0][0] * original_vertices_range[1])
    f = f.translate(original_vertices_center)

    if verbose:
        return f, triviews_all
    return f

def normalize_mesh(verts, faces, render_scale):
    original_mesh = trimesh.Trimesh(vertices=verts, faces=faces)

    original_vertices_range = original_mesh.vertices.ptp(axis=0)
    original_center = original_vertices_range / 2 + original_mesh.vertices.min(axis=0)

    normalized_mesh = original_mesh.copy()
    normalized_mesh = normalized_mesh.apply_translation(-original_center)
    normalized_mesh = normalized_mesh.apply_scale(1.0 / original_vertices_range.max() * 2 * render_scale)

    normalized_vertices_range = normalized_mesh.vertices.ptp(axis=0)
    normalized_verts_height = normalized_vertices_range[1]
    return normalized_mesh, normalized_verts_height, 


def sdf_to_trimesh(f, step=0.01):
    triangle_faces = f.generate(step=step)
    verts, faces = np.unique(triangle_faces, axis=0, return_inverse=True)
    faces = faces.reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    return mesh


def render_mesh_by_pyrender(mesh, width=1000, height=1000):
    import pyrender

    r = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    scene = pyrender.Scene()
    prmesh = pyrender.Mesh.from_trimesh(mesh)
    scene.add(prmesh, pose=np.eye(4))

    camera = pyrender.OrthographicCamera(xmag=1.0, ymag=1.0)
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 30.0],
        [0.0, 0.0, 0.0, 1.0],
        ])
    scene.add(camera, pose=camera_pose)
    _, depth = r.render(scene)
    r.delete()

    return depth

def render_mesh_by_nvdiffrast(nvdiffrast_context, mesh, width=1000, height=1000):
    import nvdiffrast.torch as dr
    import torch
    verts = torch.from_numpy(mesh.vertices).float().cuda()
    verts[:, 1] *= -1
    verts = torch.nn.functional.pad(verts, (0, 1), value=1.0)[None]
    faces = torch.from_numpy(mesh.faces).int().cuda()
    with torch.no_grad():
        rast, _ = dr.rasterize(nvdiffrast_context, verts, faces, (int(height), int(width)))
    return rast[0, ..., -1].cpu().numpy()

@click.command()
@click.option("--mesh_file", "-i", type=str, default="mesh.obj")
@click.option("--mesh_bbox_file", "-o", type=str, default="mesh_bbox.obj")
@click.option("--edge_process", "-e", type=click.Choice(["dilate", "erode", "none"]), default="none")
@click.option("--padding", "-p", type=float, default=0.02)
@click.option("--nvdiffrast_context", "-n", is_flag=True)
@click.option("--debug", is_flag=True)
def main(mesh_file, mesh_bbox_file, edge_process, padding, nvdiffrast_context, debug):
    logging.basicConfig(level=logging.INFO, force=True)
    original_mesh = trimesh.load_mesh(mesh_file)
    if nvdiffrast_context is True:
        import nvdiffrast.torch as dr
        nvdiffrast_context = dr.RasterizeGLContext()
    else:
        nvdiffrast_context = None
    f, triviews_all = get_mesh_bbox_sdf(original_mesh.vertices, original_mesh.faces, edge_process, padding_percent=padding, nvdiffrast_context=nvdiffrast_context, debug=debug, verbose=True)
    mesh = sdf_to_trimesh(f)
    _ = trimesh.exchange.export.export_mesh(mesh, mesh_bbox_file)
    for i, triview in enumerate(triviews_all):
        cv2.imwrite(os.path.join(os.path.dirname(mesh_bbox_file), f"{i}.png"), triview)

if __name__ == "__main__":
    main()
