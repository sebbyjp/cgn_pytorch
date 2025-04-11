from typing import overload, cast
import meshcat
import meshcat.geometry as g
import numpy as np
import os
import argparse
from trimesh import exchange, Scene
from trimesh.points import concatenate, apply_transform, scale_matrix
from importlib.resources import files
from mbodios.geometry.array import array
from mbodios.types.ndarray import N, sz,Float

def scale_matrix(factor:float|array[sz[3],Float], origin:array[sz[3],Float]|None=None):
    """Return matrix to scale by factor around origin in direction.
    Use factor -1 for point symmetry.
    """
    if not isinstance(factor, list) and not isinstance(factor, np.ndarray):
        M = np.diag([factor, factor, factor, 1.0])
    else:
        assert len(factor) == 3, 'If applying different scaling per dimension, must pass in 3-element list or array'
        #M = np.diag([factor[0], factor[1], factor[2], 1.0])
        M = np.eye(4)
        M[0, 0] = factor[0]
        M[1, 1] = factor[1]
        M[2, 2] = factor[2]
    if origin is not None:
        M[:3, 3] = origin[:3]
        M[:3, 3] *= 1.0 - factor
    return M
@overload
def meshcat_pcd_show(mc_vis, point_cloud:array[N,sz[3],Float],color:array[sz[3],N,Float]|None=None, name:str|None=None, size:float=0.001):...
@overload
def meshcat_pcd_show(mc_vis, point_cloud:array[sz[3],N,Float],color:array[sz[3],N,Float]|None=None, name:str|None=None, size:float=0.001):...
def meshcat_pcd_show(mc_vis, point_cloud:array[N,sz[3],Float] | array[sz[3],N,Float], color:array[sz[3],N,Float]|array[sz[3],Float]|None=None, name:str|None=None, size:float=0.001):
    """
    Function to show a point cloud using meshcat. 
    mc_vis (meshcat.Visualizer): Interface to the visualizer 
    point_cloud (np.ndarray): Shape Nx3 or 3xN
    color (np.ndarray or list): Shape (3,)
    """
    if point_cloud.shape[0] != 3:
        point_cloud = cast(array[sz[3],N,Float], point_cloud)
        point_cloud = point_cloud.transpose(axes=(1,0))
    if color is None:
        color_pts = np.zeros_like(point_cloud) * 255
    else:
        color_pts = np.zeros_like(point_cloud)
        color_pts[0,:] = color[0]*255
        color_pts[1,:] = color[1]*255
        color_pts[2,:] = color[2]*255
    if name is None:
        name = 'scene/pcd'

    mc_vis[name].set_object(
        g.Points(
            g.PointsGeometry(point_cloud, color=color_pts),
            g.PointsMaterial(size=size)
    ))


def show_mesh(vis, paths, poses, scales, names, clear=False,  opacity=1.0, color=(128,128,128)):
    if vis is None:
        vis = meshcat.Visualizer(zmq_url='tcp://127.0.0.1:6000')
    if clear:
        vis['scene/'].delete()
        vis['home/'].delete()
        vis.delete()
    color = int('%02x%02x%02x' % color, 16)

    for i, (path, pose, scale, name) in enumerate(zip(paths, poses, scales, names)):
        name = name.split('/')[-1]
        trimesh_mesh = exchange.load.load(path, file_type='obj')
        if type(trimesh_mesh) == Scene:
            meshes = []
            for geometry in trimesh_mesh.geometry:
                meshes.append(trimesh_mesh.geometry[geometry])
            trimesh_mesh = concatenate(meshes)
        scale_tf = scale_matrix(scale)
        trimesh_mesh = apply_transform(scale_tf)
        trimesh_mesh.vertices -= np.mean(trimesh_mesh.vertices, 0)
        
        trimesh_mesh = apply_transform(pose)

        verts = trimesh_mesh.vertices
        faces = trimesh_mesh.faces
        
        material = meshcat.geometry.MeshLambertMaterial(color=color, reflectivity=0.0, opacity=opacity)
        mcg_mesh = meshcat.geometry.TriangularMeshGeometry(verts, faces)
        vis['scene/'+name].set_object(mcg_mesh, material)
    return vis

    
def sample_grasp_show(mc_vis, control_pt_list:array[N,sz[3],Float] | array[sz[3],N,Float], name=None, freq=100):
    """
    shows a sample grasp as represented by a little fork guy
    freq: show one grasp per every (freq) grasps (1/freq is ratio of visualized grasps)
    """
    if name is None:
        name = 'scene/loop/'
    for i, gripper in enumerate(control_pt_list):
        # color = np.zeros_like(gripper) * 255
        # wrist = gripper[[1, 0, 2], :]
        # wrist = np.transpose(wrist, axes=(1,0))
        
        gripper = gripper[1:,:]
        gripper = gripper[[2, 0, 1, 3], :]
        gripper = gripper.transpose(axes=(1,0))
        
        name_i = 'pose'+str(i)
        if i%freq == 0:
            mc_vis[name+name_i].set_object(g.Line(g.PointsGeometry(gripper)))
            #mc_vis[name_i+'wrist'].set_object(g.Line(g.PointsGeometry(wrist)))

def mesh_gripper(mc_vis, pose, name=None, robotiq=False):
    resource_path = str(files('cgn_pytorch').joinpath('grippers/mesh'))
    print('\n\n\n resource: \n\n',resource_path)
    if robotiq:
        gripper_path = os.path.join(resource_path, '/robotiq_arg2f_base_link.stl')
    else:
        gripper_path = os.path.join(resource_path, 'panda_gripper/panda_gripper.obj')
    gripper = meshcat.geometry.ObjMeshGeometry.from_file(gripper_path)
    if name is None:
        name = 'gripper'
    mc_vis['scene/'+name].set_object(gripper)
    # print('pose: ', pose)
    #mc_vis['scene/'+name].set_transform(pose)
    mc_vis['scene/'+name].set_transform(pose.astype(np.float64))
            
def viz_pcd(np_pc, name, grasps=False, gripper=False, robotiq=False, clear=False, freq=1):
    vis = meshcat.Visualizer(zmq_url='tcp://127.0.0.1:6000')
    #print('MeshCat URL: %s' % vis.url())
    # vis['scene/'+name].delete()
    if clear:
        vis['scene'].delete()
        vis.delete()
    if grasps:
        sample_grasp_show(vis, np_pc, name=name, freq=freq)
    elif gripper:
        mesh_gripper(vis, np_pc, name=name, robotiq=robotiq)
    else:
        meshcat_pcd_show(vis, np_pc, name=name)


def viz_scene(vis, paths, poses, scales, names, cmeans=None, clear=False, opacity=1.0, goal=False):
    if vis is None:
        vis = meshcat.Visualizer(zmq_url='tcp://127.0.0.1:6000')
    if clear:
        vis['scene/'].delete()
        vis['home/'].delete()
        vis.delete()
    for i, (path, pose, scale, name) in enumerate(zip(paths, poses, scales, names)):
        name = name.split('/')[-1]
        if goal:
            name = 'goal/' + name
        print('path:', path)
        trimesh_mesh = exchange.load.load(path, file_type=path.split('.')[-1])
        if type(trimesh_mesh) == trimesh.Scene:
            meshes = []
            for geometry in trimesh_mesh.geometry:
                meshes.append(trimesh_mesh.geometry[geometry])
            trimesh_mesh = trimesh.util.concatenate(meshes)

        scale_tf = trimesh.transformations.scale_matrix(scale)
        trimesh_mesh = trimesh_mesh.apply_transform(scale_tf)
        if cmeans is None:
            trimesh_mesh.vertices -= np.mean(trimesh_mesh.vertices,0) #trimesh_mesh.center_mass
        else:
            trimesh_mesh.vertices -= cmeans[i]
        trimesh_mesh = trimesh_mesh.apply_transform(pose)

        verts = trimesh_mesh.vertices
        faces = trimesh_mesh.faces

        color=(128, 128, 128)
        color = int('%02x%02x%02x' % color, 16)

        material = meshcat.geometry.MeshLambertMaterial(color=color, reflectivity=0.0, opacity=opacity)
        mcg_mesh = meshcat.geometry.TriangularMeshGeometry(verts, faces)
        vis['scene/'+name].set_object(mcg_mesh, material)
        # print('pose: ', pose)
    return vis
        
            
def visualize(args):
    vis = meshcat.Visualizer(zmq_url='tcp://127.0.0.1:6000')
    vis['scene'].delete()
    vis.delete()
    print('MeshCat URL: %s' % vis.url())

    # pb_pcd = np.load('pybullet_pcd.npy')
    
    cam_pose = np.load('cam_pose.npy')
    box = meshcat.geometry.Box([0.1, 0.2, 0.3])
    vis['scene/cam'].set_object(box)
    vis['scene/cam'].set_transform(cam_pose)
    
    threshold = 0.2
    pred_s = np.load('pred_s_mask.npy')
    print(np.max(pred_s))
    success_mask = np.where(pred_s > threshold)
    pcd = np.load('full_pcd.npy')
    s_pcd = pcd[success_mask]
    print(success_mask)
    green = np.zeros_like(s_pcd)
    green[:, 1] = 255*np.ones_like(s_pcd)[:,1]

    pc_world = np.load('world_pc.npy')
    # pc_cam = np.load('cam_pc.npy')
    if args.i is not None:
        grasps = np.load('control_pt_list.npy')[:args.i]
        grasp_labels = np.load('label_pt_list.npy')[:args.i]
    else:
        grasps = np.load('control_pt_list.npy')
        grasp_labels = np.load('label_pt_list.npy')
        
    print('pred', grasps.shape)
    pc_gt = np.load('ground_truth.npy')
    print('labels', grasp_labels.shape)

    pose = np.eye(4)
    pose[:3, :3] = pcd[0]

    #meshcat_pcd_show(vis, pb_pcd, name='scene/pb')

    meshcat_pcd_show(vis, pc_world, name='scene/world')
    print('show world pc')
    meshcat_pcd_show(vis, pc_gt, name='scene/gt')
    meshcat_pcd_show(vis, s_pcd, name='scene/s_pcd', color=green.T)
    sample_grasp_show(vis, grasps, name='pred/', freq=1)

    d = np.load('d.npy')
    meshcat_pcd_show(vis, d, name='scene/d')

    
    '''
    obs_color = np.zeros_like(pos_pcd)
    obs_color[:, 0] = 255*np.ones_like(pos_pcd)[:, 0]
    green_color = np.zeros_like(pos_labeled)
    green_color[:, 1] = 255*np.ones_like(pos_labeled)[:, 1]
    white_color = 255*np.ones_like(pred_s_pcd)
    '''
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--single_grasp', type=bool, default=False, help='load a single grasp with contact point emphasized')
    parser.add_argument('--i', type=int, default=None, help='index for single grasp viz')
    parser.add_argument('--demo', type=bool, default=False, help='run the hierarchical scene demo')
    args = parser.parse_args()
    
    if args.demo:
        run_hierarchical_demo()
    else:
        visualize(args)


def run_hierarchical_demo():
    """
    Demonstrates a hierarchical scene graph with transformation caching and Merkle-style 
    content addressing for efficient visualization and updates.
    """
    import time
    import hashlib
    from collections import defaultdict
    import trimesh
    
    print("Starting hierarchical scene graph demo with caching...")
    vis = meshcat.Visualizer(zmq_url='tcp://127.0.0.1:6000')
    vis['scene'].delete()
    vis.delete()
    print(f'MeshCat URL: {vis.url()}')
    
    # Cache for storing mesh data by content hash
    mesh_cache = {}
    # Cache for storing transforms
    transform_cache = {}
    # Node hierarchy
    scene_hierarchy = {}
    # Track mesh usage for garbage collection
    mesh_usage = defaultdict(int)
    
    def content_hash(data):
        """Generate a content hash for mesh data"""
        return hashlib.md5(data.tobytes()).hexdigest()
    
    def add_hierarchical_object(name, mesh, parent=None, transform=None):
        """Add object to scene with hierarchical transforms and content-based caching"""
        # Get or create content hash for efficient storage
        mesh_bytes = mesh.vertices.tobytes() + mesh.faces.tobytes()
        mesh_id = content_hash(mesh_bytes)
        
        # Track usage for this mesh
        mesh_usage[mesh_id] += 1
        
        # Store in cache if not already present
        if mesh_id not in mesh_cache:
            print(f"Caching new mesh {mesh_id}")
            mesh_cache[mesh_id] = mesh
        
        # Set up hierarchy
        full_path = f"{parent}/{name}" if parent else name
        scene_hierarchy[full_path] = {
            'mesh_id': mesh_id,
            'parent': parent,
            'children': []
        }
        
        if parent and parent in scene_hierarchy:
            scene_hierarchy[parent]['children'].append(full_path)
        
        # Set transform
        if transform is not None:
            transform_cache[full_path] = transform
        
        # Create visualization objects
        verts = mesh.vertices
        faces = mesh.faces
        
        # Visualize
        material = meshcat.geometry.MeshLambertMaterial(reflectivity=0.0)
        mcg_mesh = meshcat.geometry.TriangularMeshGeometry(verts, faces)
        vis_path = f'scene/{full_path}'
        vis[vis_path].set_object(mcg_mesh, material)
        
        # Apply transform if available
        if transform is not None:
            vis[vis_path].set_transform(transform.astype(np.float64))
            
        return full_path
    
    def update_hierarchical_transform(name, new_transform):
        """Update transform of an object and propagate to children"""
        if name not in scene_hierarchy:
            print(f"Object {name} not found in scene")
            return
        
        # Update transform in cache
        transform_cache[name] = new_transform
        
        # Apply to visualization
        vis_path = f'scene/{name}'
        vis[vis_path].set_transform(new_transform.astype(np.float64))
        
        # Recursively update children
        update_children(name, np.eye(4))
        
    def update_children(parent, parent_transform):
        """Recursively update children transforms"""
        for child in scene_hierarchy[parent]['children']:
            # Compute child's world transform
            child_local = transform_cache.get(child, np.eye(4))
            child_world = parent_transform @ child_local
            
            # Apply
            vis_path = f'scene/{child}'
            vis[vis_path].set_transform(child_world.astype(np.float64))
            
            # Recurse
            update_children(child, child_world)
    
    # Create a simple hierarchical scene
    print("Creating hierarchical scene...")
    
    # Base object
    box = trimesh.creation.box((0.5, 0.5, 0.1))
    base_tf = np.eye(4)
    base_name = add_hierarchical_object("base", box, transform=base_tf)
    
    # Add a cylinder on top
    cylinder = trimesh.creation.cylinder(radius=0.1, height=0.3)
    cyl_tf = np.eye(4)
    cyl_tf[2, 3] = 0.2  # Move up
    cyl_name = add_hierarchical_object("cylinder", cylinder, parent=base_name, transform=cyl_tf)
    
    # Add a sphere on top of cylinder
    sphere = trimesh.creation.icosphere(radius=0.1)
    sphere_tf = np.eye(4)
    sphere_tf[2, 3] = 0.3  # Move up relative to parent
    sphere_name = add_hierarchical_object("sphere", sphere, parent=cyl_name, transform=sphere_tf)
    
    # Add identical spheres elsewhere to demonstrate caching
    sphere2_tf = np.eye(4)
    sphere2_tf[0, 3] = 0.3  # Move along x
    add_hierarchical_object("sphere2", sphere, transform=sphere2_tf)
    
    # Verify same mesh is reused
    mesh_id = scene_hierarchy[sphere_name]['mesh_id']
    print(f"Sphere mesh ID: {mesh_id}")
    print(f"Mesh usage count: {mesh_usage[mesh_id]}")
    print(f"Cache size: {len(mesh_cache)} unique meshes")
    
    # Animation loop
    print("Running animation...")
    try:
        for i in range(100):
            # Rotate the base, all children will update
            angle = i * 0.05
            rot = np.eye(4)
            rot[0:3, 0:3] = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])[0:3, 0:3]
            
            # Update transform - this automatically propagates to children
            update_hierarchical_transform(base_name, rot)
            
            # Small delay
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("Animation stopped by user")
    
    print("Demo complete!")
