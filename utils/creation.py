import trimesh
import scipy
import numpy as np
import torch
import distinctipy


def create_bone_capsule(heads, tails, radius=0.08, add_color=False):
    if isinstance(heads, torch.Tensor):
        heads = heads.cpu().numpy()
    if isinstance(tails, torch.Tensor):
        tails = tails.cpu().numpy()

    mids = 0.5 * (heads + tails)
    dirs = tails - heads
    dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
    init_dirs = np.zeros_like(dirs)
    init_dirs[:, -1] = 1.0

    meshes = []

    if add_color:
        colors = distinctipy.get_colors(len(heads), pastel_factor=0.5, rng=1)
        colors = np.array(colors) * 255
        colors = colors.astype(np.uint8)

    for i in range(heads.shape[0]):
        head, tail = heads[i], tails[i]
        height = np.linalg.norm(tail - head)

        mesh = trimesh.creation.capsule(radius=radius, height=height)
        mesh_verts = mesh.vertices
        mesh_verts[:, 2] -= mesh_verts[:, 2].mean()  # center the mesh along z-axis

        rot = scipy.spatial.transform.Rotation.align_vectors(init_dirs[i:i+1], dirs[i:i+1])[0].as_matrix()
        mesh_verts = np.matmul(mesh_verts, rot) + mids[i]
        mesh.vertices = mesh_verts

        if add_color:
            mesh.visual.vertex_colors[:, :3] = colors[i]

        meshes.append(mesh)
    
    return trimesh.util.concatenate(meshes)