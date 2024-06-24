import open3d as o3d
import numpy as np
import trimesh
from typing import Optional, List
import matplotlib.pyplot as plt


def trimeshToO3D(mesh:trimesh, rgb:Optional[list]= None, points:Optional[bool]= False):
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    mesh_o3d= o3d.geometry.TriangleMesh()
    mesh_o3d.vertices=o3d.utility.Vector3dVector(vertices)
    mesh_o3d.triangles=o3d.utility.Vector3iVector(faces)
    mesh_o3d.compute_vertex_normals()
    if rgb is None:
        # compute rgb colour based on normals
        vex_color_rgb = np.abs(mesh_o3d.vertex_normals)
    else:
        vex_color_rgb = np.ones([mesh.vertices.shape[0],3]) * rgb
    mesh_o3d.vertex_colors=o3d.utility.Vector3dVector(vex_color_rgb)

    if points == True:
        mesh_o3d= o3d.geometry.PointCloud()
        mesh_o3d.points = o3d.utility.Vector3dVector(mesh.vertices)
    return mesh_o3d

def visualize(mesh):
    o3d.visualization.draw_geometries([mesh])

def turntable(mesh, save:Optional[bool]=True):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Visualizer')
    vis.get_render_option().background_color = np.asarray([0, 0, 0])
    
    # move the mesh back to fit everything in frame
    
    vis.add_geometry(mesh)
    mesh = mesh.translate([0, 0, -0.05])
    rot = 8
    for i in range(int(360/rot)):
        # rotate the mesh by 1 degree
        mesh = mesh.rotate(mesh.get_rotation_matrix_from_xyz((0, rot*(np.pi/180), 0)), center=mesh.get_center())
        vis.update_geometry(mesh)
        vis.poll_events()
        vis.update_renderer()
        if save:
            vis.capture_screen_image(f"turntable/{i}.png")

    vis.destroy_window()

def viz_3d(mesh: trimesh.Trimesh,
           intermediate_points: List[np.ndarray],
           RES_PATH: str,
           image_index: Optional[int] = 0,
           save_image: Optional[bool] = True):
    
    # Convert mesh to Open3D format
    initial = trimeshToO3D(mesh, [1, 0, 0])
    
    # Initialize Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(initial)

    for i, update_points in enumerate(intermediate_points):
        # ensure the update_point is float64
        update_points = update_points.astype(np.float64)
        initial.vertices = o3d.utility.Vector3dVector(update_points)
        initial.compute_vertex_normals()
        vis.update_geometry(initial)
        if save_image:
            image = vis.capture_screen_float_buffer()
            plt.imsave(f"{RES_PATH}/{i + image_index}.png", np.asarray(image))
    vis.run()
    vis.destroy_window()

def trans_trimesh(tm:trimesh, 
                  angle:int, 
                  shift:Optional[int]=None):
    # rotation
    rad = angle / 180 * np.pi
    R = np.array([[np.cos(rad), 0, np.sin(rad)],
                  [0, 1, 0],
                  [-np.sin(rad), 0, np.cos(rad)]])
    homo_R = np.eye(4)
    homo_R[:3, :3] = R

    # shift
    if shift is None:
        homo_T = np.eye(4)
    else:
        centroid = np.mean(np.array(tm.vertices), axis=0)
        homo_T = np.eye(4)
        homo_T[:3, 3] = -centroid + np.array(shift)

    tm.apply_transform(homo_R @ homo_T)