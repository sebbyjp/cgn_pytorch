import os
import glob
import numpy as np
from scipy.spatial import KDTree, cKDTree
from cgn_pytorch.util.scene_renderer import SceneRenderer
import trimesh.transformations as tra
import copy
import torch
from scipy import spatial

from mbodios.types.sense.image import Image

def get_obj_surrounding(pcd, obj_mask, radius):
    '''
    gets points within a distance of an object in scene pointcloud
    arguments:
        pcd -- full pointcloud of entire cluttered scene
        obj_mask -- segmentation mask of target object
        radius -- max distance of points from any point in the target object
    returns:
        surround_mask -- mask of desired surrounding points
    '''
    obj_mask = obj_mask.view(-1).detach().cpu().numpy()
    surround = pcd[np.logical_not(obj_mask), :]
    tree = spatial.KDTree(surround)
    neighbor_idxs = tree.query_ball_point(pcd[obj_mask, :], radius)
    flat_neighbors = []
    for a in neighbor_idxs:
        flat_neighbors += a
    surround_mask = np.unique(flat_neighbors)

    # del(flat_neighbors)
    # del(neighbor_idxs)
    # gc.collect()

    return surround_mask

def load_scene_contacts(dataset_folder, test_split_only=False, num_test=None, scene_contacts_path='scene_contacts_new'):
    """
    Load contact grasp annotations from acronym scenes 
    Arguments:
        dataset_folder {str} -- folder with acronym data and scene contacts
    Keyword Arguments:
        test_split_only {bool} -- whether to only return test split scenes (default: {False})
        num_test {int} -- how many test scenes to use (default: {None})
        scene_contacts_path {str} -- name of folder with scene contact grasp annotations (default: {'scene_contacts_new'})
    Returns:
        list(dicts) -- list of scene annotations dicts with object paths and transforms and grasp contacts and transforms.
    """
    scene_contact_paths = sorted(glob.glob(os.path.join(dataset_folder, scene_contacts_path, '*')))
    if test_split_only:
        scene_contact_paths = scene_contact_paths[-num_test:]
    contact_infos = []
    for contact_path in scene_contact_paths:
        #print(contact_path)
        try:
            npz = np.load(contact_path, allow_pickle=True, encoding='bytes')
            contact_info = {'scene_contact_points':npz['scene_contact_points'],
                            'obj_paths':npz['obj_paths'],
                            'obj_transforms':npz['obj_transforms'],
                            'obj_scales':npz['obj_scales'],
                            'grasp_transforms':npz['grasp_transforms']}
            contact_infos.append(contact_info)
        except:
            print('corrupt, ignoring..')
    return contact_infos

def regularize_pc_point_count(pc, npoints, use_farthest_point=False):
    """
      If point cloud pc has less points than npoints, it oversamples.
      Otherwise, it downsample the input pc to have npoint points.
      use_farthest_point: indicates 
      
      :param pc: Nx3 point cloud
      :param npoints: number of points the regularized point cloud should have
      :param use_farthest_point: use farthest point sampling to downsample the points, runs slower.
      :returns: npointsx3 regularized point cloud
    """
    
    if pc.shape[0] > npoints:
        if use_farthest_point:
            _, center_indexes = farthest_points(pc, npoints, distance_by_translation_point, return_center_indexes=True)
        else:
            center_indexes = np.random.choice(range(pc.shape[0]), size=npoints, replace=False)
        pc = pc[center_indexes, :]
    else:
        required = npoints - pc.shape[0]
        if required > 0:
            index = np.random.choice(range(pc.shape[0]), size=required)
            pc = np.concatenate((pc, pc[index, :]), axis=0)
    return pc

def preprocess_pc_for_inference(input_pc, num_point, pc_mean=None, return_mean=False, use_farthest_point=False, convert_to_internal_coords=False):
    """
    Various preprocessing of the point cloud (downsampling, centering, coordinate transforms)  
    Arguments:
        input_pc {np.ndarray} -- Nx3 input point cloud
        num_point {int} -- downsample to this amount of points
    Keyword Arguments:
        pc_mean {np.ndarray} -- use 3x1 pre-computed mean of point cloud  (default: {None})
        return_mean {bool} -- whether to return the point cloud mean (default: {False})
        use_farthest_point {bool} -- use farthest point for downsampling (slow and suspectible to outliers) (default: {False})
        convert_to_internal_coords {bool} -- Convert from opencv to internal coordinates (x left, y up, z front) (default: {False})
    Returns:
        [np.ndarray] -- num_pointx3 preprocessed point cloud
    """
    normalize_pc_count = input_pc.shape[0] != num_point
    if normalize_pc_count:
        pc = regularize_pc_point_count(input_pc, num_point, use_farthest_point=use_farthest_point).copy()
    else:
        pc = input_pc.copy()
    
    if convert_to_internal_coords:
        pc[:,:2] *= -1

    if pc_mean is None:
        pc_mean = np.mean(pc, 0)

    pc -= np.expand_dims(pc_mean, 0)
    if return_mean:
        return pc, pc_mean
    else:
        return pc


def inverse_transform(trans):
    """
    Computes the inverse of 4x4 transform.
    Arguments:
        trans {np.ndarray} -- 4x4 transform.
    Returns:
        [np.ndarray] -- inverse 4x4 transform
    """
    rot = trans[:3, :3]
    t = trans[:3, 3]
    rot = np.transpose(rot)
    t = -np.matmul(rot, t)
    output = np.zeros((4, 4), dtype=np.float32)
    output[3][3] = 1
    output[:3, :3] = rot
    output[:3, 3] = t

    return output

def farthest_points(data, nclusters, return_center_indexes=False, return_distances=False, verbose=False):
    """
      Performs farthest point sampling on data points.
      Args:
        data: numpy array of the data points.
        nclusters: int, number of clusters.
        dist_dunc: distance function that is used to compare two data points.
        return_center_indexes: bool, If True, returns the indexes of the center of 
          clusters.
        return_distances: bool, If True, return distances of each point from centers.
      
      Returns clusters, [centers, distances]:
        clusters: numpy array containing the cluster index for each element in 
          data.
        centers: numpy array containing the integer index of each center.
        distances: numpy array of [npoints] that contains the closest distance of 
          each point to any of the cluster centers.
    """
    if nclusters >= data.shape[0]:
        if return_center_indexes:
            return np.arange(data.shape[0], dtype=np.int32), np.arange(data.shape[0], dtype=np.int32)

        return np.arange(data.shape[0], dtype=np.int32)

    clusters = np.ones((data.shape[0],), dtype=np.int32) * -1
    distances = np.ones((data.shape[0],), dtype=np.float32) * 1e7
    centers = []
    for iter in range(nclusters):
        index = np.argmax(distances)
        centers.append(index)
        shape = list(data.shape)
        for i in range(1, len(shape)):
            shape[i] = 1

        broadcasted_data = np.tile(np.expand_dims(data[index], 0), shape)
        new_distances = np.linalg.norm(broadcasted_data - data)
        distances = np.minimum(distances, new_distances)
        clusters[distances == new_distances] = iter
        if verbose:
            print('farthest points max distance : {}'.format(np.max(distances)))

    if return_center_indexes:
        if return_distances:
            return clusters, np.asarray(centers, dtype=np.int32), distances
        return clusters, np.asarray(centers, dtype=np.int32)

    return clusters

def reject_median_outliers(data, m=0.4, z_only=False):
    """
    Reject outliers with median absolute distance m
    Arguments:
        data {[np.ndarray]} -- Numpy array such as point cloud
    Keyword Arguments:
        m {[float]} -- Maximum absolute distance from median in m (default: {0.4})
        z_only {[bool]} -- filter only via z_component (default: {False})
    Returns:
        [np.ndarray] -- Filtered data without outliers
    """
    if z_only:
        d = np.abs(data[:,2:3] - np.median(data[:,2:3]))
    else:
        d = np.abs(data - np.median(data, axis=0, keepdims=True))

    return data[np.sum(d, axis=1) < m]

def reject_z_outliers(data): #, cam, m=1):
    '''
    Reject points that arise from bug in rendering (~5000 points at the camera location)
    '''
    d = np.linalg.norm(data[:,:3], axis=1)
    return data[d > 1]

def regularize_pc_point_count(pc, npoints, use_farthest_point=False):
    """
      If point cloud pc has less points than npoints, it oversamples.
      Otherwise, it downsample the input pc to have npoint points.
      use_farthest_point: indicates 
      
      :param pc: Nx3 point cloud
      :param npoints: number of points the regularized point cloud should have
      :param use_farthest_point: use farthest point sampling to downsample the points, runs slower.
      :returns: npointsx3 regularized point cloud
    """
    
    if pc.shape[0] > npoints:
        if use_farthest_point:
            _, center_indexes = farthest_points(pc, npoints, distance_by_translation_point, return_center_indexes=True)
        else:
            center_indexes = np.random.choice(range(pc.shape[0]), size=npoints, replace=False)
        pc = pc[center_indexes, :]
    else:
        required = npoints - pc.shape[0]
        if required > 0:
            index = np.random.choice(range(pc.shape[0]), size=required)
            pc = np.concatenate((pc, pc[index, :]), axis=0)
    return pc

def depth2pc(depth, K, rgb=None):
    """
    Convert depth and intrinsics to point cloud and optionally point cloud color
    :param depth: hxw depth map in m
    :param K: 3x3 Camera Matrix with intrinsics
    :returns: (Nx3 point cloud, point cloud color)
    """

    mask = np.where(depth > 0)
    x,y = mask[1], mask[0]
    
    normalized_x = (x.astype(np.float32) - K[0,2])
    normalized_y = (y.astype(np.float32) - K[1,2])

    world_x = normalized_x * depth[y, x] / K[0,0]
    world_y = normalized_y * depth[y, x] / K[1,1]
    world_z = depth[y, x]

    if rgb is not None:
        rgb = rgb[y,x,:]
        
    pc = np.vstack((world_x, world_y, world_z)).T
    return (pc, rgb)


def estimate_normals_cam_from_pc(pc_cam, max_radius=0.05, k=12):
    """
    Estimates normals in camera coords from given point cloud.
    Arguments:
        pc_cam {np.ndarray} -- Nx3 point cloud in camera coordinates
    Keyword Arguments:
        max_radius {float} -- maximum radius for normal computation (default: {0.05})
        k {int} -- Number of neighbors for normal computation (default: {12})
    Returns:
        [np.ndarray] -- Nx3 point cloud normals
    """
    print('this was called')
    tree = cKDTree(pc_cam, leafsize=pc_cam.shape[0]+1)
    _, ndx = tree.query(pc_cam, k=k, distance_upper_bound=max_radius, n_jobs=8) # num_points x k
    
    for c,idcs in enumerate(ndx):
        idcs[idcs==pc_cam.shape[0]] = c
        ndx[c,:] = idcs
    neighbors = np.array([pc_cam[ndx[:,n],:] for n in range(k)]).transpose((1,0,2))
    pc_normals = vectorized_normal_computation(pc_cam, neighbors)
    return pc_normals

def vectorized_normal_computation(pc, neighbors):
    """
    Vectorized normal computation with numpy
    Arguments:
        pc {np.ndarray} -- Nx3 point cloud
        neighbors {np.ndarray} -- Nxkx3 neigbours
    Returns:
        [np.ndarray] -- Nx3 normal directions
    """
    diffs = neighbors - np.expand_dims(pc, 1) # num_point x k x 3
    covs = np.matmul(np.transpose(diffs, (0, 2, 1)), diffs) # num_point x 3 x 3
    covs /= diffs.shape[1]**2
    # takes most time: 6-7ms
    eigen_values, eigen_vectors = np.linalg.eig(covs) # num_point x 3, num_point x 3 x 3
    orders = np.argsort(-eigen_values, axis=1) # num_point x 3
    orders_third = orders[:,2] # num_point
    directions = eigen_vectors[np.arange(pc.shape[0]),:,orders_third]  # num_point x 3
    dots = np.sum(directions * pc, axis=1) # num_point
    directions[dots >= 0] = -directions[dots >= 0]
    return directions

def load_available_input_data(p, K=None):
    """
    Load available data from input file path. 
    
    Numpy files .npz/.npy should have keys
    'depth' + 'K' + (optionally) 'segmap' + (optionally) 'rgb'
    or for point clouds:
    'xyz' + (optionally) 'xyz_color'
    
    png files with only depth data (in mm) can be also loaded.
    If the image path is from the GraspNet dataset, corresponding rgb, segmap and intrinic are also loaded.
      
    :param p: .png/.npz/.npy file path that contain depth/pointcloud and optionally intrinsics/segmentation/rgb
    :param K: 3x3 Camera Matrix with intrinsics
    :returns: All available data among segmap, rgb, depth, cam_K, pc_full, pc_colors
    """
    
    segmap, rgb, depth, pc_full, pc_colors = None, None, None, None, None

    if K is not None:
        if isinstance(K,str):
            cam_K = eval(K)
        cam_K = np.array(K).reshape(3,3)

    if '.np' in p:
        data = np.load(p, allow_pickle=True, encoding='bytes')
        if '.npz' in p:
            keys = data.files
        else:
            keys = []
            if len(data.shape) == 0:
                data = data.item()
                keys = data.keys()
            elif data.shape[-1] == 3:
                pc_full = data
            else:
                depth = data

        if 'depth' in keys:
            depth = data['depth']
            if K is None and 'K' in keys:
                cam_K = data['K'].reshape(3,3)
            if 'segmap' in keys:    
                segmap = data['segmap']
            if 'seg' in keys:    
                segmap = data['seg']
            if 'rgb' in keys:    
                rgb = data['rgb']
                rgb = np.array(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        elif 'xyz' in keys:
            pc_full = np.array(data['xyz']).reshape(-1,3)
            if 'xyz_color' in keys:
                pc_colors = data['xyz_color']
    elif '.png' in p:
        if os.path.exists(p.replace('depth', 'label')):
            # graspnet data
            depth, rgb, segmap, K = load_graspnet_data(p)
        elif os.path.exists(p.replace('depths', 'images').replace('npy', 'png')):
            rgb = np.array(Image.open(p.replace('depths', 'images').replace('npy', 'png')))
        else:
            depth = np.array(Image.open(p))
    else:
        raise ValueError('{} is neither png nor npz/npy file'.format(p))
    
    return segmap, rgb, depth, cam_K, pc_full, pc_colors

def load_graspnet_data(rgb_image_path):
    
    """
    Loads data from the GraspNet-1Billion dataset
    # https://graspnet.net/
    :param rgb_image_path: .png file path to depth image in graspnet dataset
    :returns: (depth, rgb, segmap, K)
    """
    
    depth = Image.open(rgb_image_path).array/1000. # m to mm
    segmap = Image.open(rgb_image_path.replace('depth', 'label')))
    rgb = Image.open(rgb_image_path.replace('depth', 'rgb')))

    # graspnet images are upside down, rotate for inference
    # careful: rotate grasp poses back for evaluation
    depth = np.rot90(depth,2)
    segmap = np.rot90(segmap,2)
    rgb = np.rot90(rgb,2)
    
    if 'kinect' in rgb_image_path:
        # Kinect azure:
        K=np.array([[631.54864502 ,  0.    ,     638.43517329],
                    [  0.    ,     631.20751953, 366.49904066],
                    [  0.    ,       0.    ,       1.        ]])
    else:
        # Realsense:
        K=np.array([[616.36529541 ,  0.    ,     310.25881958],
                    [  0.    ,     616.20294189, 236.59980774],
                    [  0.    ,       0.    ,       1.        ]])

    return depth, rgb, segmap, K

def center_pc_convert_cam(cam_poses, batch_data):
    """
    Converts from OpenGL to OpenCV coordinates, computes inverse of camera pose and centers point cloud
    
    :param cam_poses: (bx4x4) Camera poses in OpenGL format
    :param batch_data: (bxNx3) point clouds 
    :returns: (cam_poses, batch_data) converted
    """
    # OpenCV OpenGL conversion
    for j in range(len(cam_poses)):
        cam_poses[j,:3,1] = -cam_poses[j,:3,1]
        cam_poses[j,:3,2] = -cam_poses[j,:3,2]
        cam_poses[j] = inverse_transform(cam_poses[j])

    pc_mean = np.mean(batch_data, axis=1, keepdims=True)
    batch_data[:,:,:3] -= pc_mean[:,:,:3]
    cam_poses[:,:3,3] -= pc_mean[:,0,:3]
    
    return cam_poses, batch_data

def load_contact_grasps(contact_list, data_config):
    """
    Loads fixed amount of contact grasp data per scene into tf CPU/GPU memory
    Arguments:
        contact_infos {list(dicts)} -- Per scene mesh: grasp contact information  
        data_config {dict} -- data config
    Returns:
        [tf_pos_contact_points, tf_pos_contact_dirs, tf_pos_contact_offsets, 
        tf_pos_contact_approaches, tf_pos_finger_diffs, tf_scene_idcs, 
        all_obj_paths, all_obj_transforms] -- tf.constants with per scene grasp data, object paths/transforms in scene
    """

    num_pos_contacts = data_config['labels']['num_pos_contacts']


    for i,c in enumerate(contact_list): # for each batch
        # embed()
        pos_contact_points = []
        pos_contact_dirs = []
        pos_finger_diffs = []
        pos_approach_dirs = []
        pos_grasp_transforms = []

        contact_directions_01 = c['scene_contact_points'][:,0,:] - c['scene_contact_points'][:,1,:]
        all_contact_points = c['scene_contact_points'].reshape(-1,3)
        all_finger_diffs = np.maximum(np.linalg.norm(contact_directions_01,axis=1), np.finfo(np.float32).eps)
        all_contact_directions = np.empty((contact_directions_01.shape[0]*2, contact_directions_01.shape[1],))
        all_contact_directions[0::2] = -contact_directions_01 / all_finger_diffs[:,np.newaxis]
        all_contact_directions[1::2] = contact_directions_01 / all_finger_diffs[:,np.newaxis]
        all_contact_suc = np.ones_like(all_contact_points[:,0])
        all_grasp_transform = c['grasp_transforms'].reshape(-1,4,4)
        all_approach_directions = all_grasp_transform[:,:3,2]

        pos_idcs = np.where(all_contact_suc>0)[0]
        if len(pos_idcs) == 0:
            continue
        all_pos_contact_points = all_contact_points[pos_idcs]
        all_pos_finger_diffs = all_finger_diffs[pos_idcs//2]
        all_pos_contact_dirs = all_contact_directions[pos_idcs]
        all_pos_approach_dirs = all_approach_directions[pos_idcs//2]
        all_grasp_transform = all_grasp_transform[pos_idcs//2]

        # Use all positive contacts then mesh_utils with replacement
        if num_pos_contacts > len(all_pos_contact_points)/2:
            pos_sampled_contact_idcs = np.arange(len(all_pos_contact_points))
            pos_sampled_contact_idcs_replacement = np.random.choice(np.arange(len(all_pos_contact_points)), num_pos_contacts*2 - len(all_pos_contact_points) , replace=True) 
            pos_sampled_contact_idcs= np.hstack((pos_sampled_contact_idcs, pos_sampled_contact_idcs_replacement))
        else:
            pos_sampled_contact_idcs = np.random.choice(np.arange(len(all_pos_contact_points)), num_pos_contacts*2, replace=False)
        pos_contact_points.append(all_pos_contact_points[pos_sampled_contact_idcs,:])
        pos_contact_dirs.append(all_pos_contact_dirs[pos_sampled_contact_idcs,:])
        pos_finger_diffs.append(all_pos_finger_diffs[pos_sampled_contact_idcs])
        pos_approach_dirs.append(all_pos_approach_dirs[pos_sampled_contact_idcs])
        pos_grasp_transforms.append(all_grasp_transform[pos_sampled_contact_idcs,:])

    #embed()

    scene_idcs = np.arange(0,len(pos_contact_points))
    contact_points = np.array(pos_contact_points)
    grasp_poses = np.array(pos_grasp_transforms)
    contact_dirs = np.array(pos_contact_dirs)
    finger_diffs = np.array(pos_finger_diffs)
    contact_approaches =  np.array(pos_approach_dirs)
        
    return contact_points, grasp_poses, contact_dirs, contact_approaches, finger_diffs, scene_idcs

def load_contact_grasps_aux(contact_list, data_config):
    """
    Loads fixed amount of contact grasp data per scene into tf CPU/GPU memory
    Arguments:
        contact_infos {list(dicts)} -- Per scene mesh: grasp contact information  
        data_config {dict} -- data config
    Returns:
        [tf_pos_contact_points, tf_pos_contact_dirs, tf_pos_contact_offsets, 
        tf_pos_contact_approaches, tf_pos_finger_diffs, tf_scene_idcs, 
        all_obj_paths, all_obj_transforms] -- tf.constants with per scene grasp data, object paths/transforms in scene
    """

    num_pos_contacts = data_config['labels']['num_pos_contacts']

    batch_pos_contact_points = []
    batch_pos_contact_dirs = []
    batch_pos_finger_diffs = []
    batch_pos_approach_dirs = []
    batch_pos_grasp_transforms = []
    
    for i,c in enumerate(contact_list): # for each batch
        # embed()
        pos_contact_points = []
        pos_contact_dirs = []
        pos_finger_diffs = []
        pos_approach_dirs = []
        pos_grasp_transforms = []
        
        for contacts, tfs, colliding in zip(c['scene_contact_points'], c['grasp_transforms'], c['var_dict'].item()['collision_labels']): # for each subgoal
            if not colliding:
                # print('getting contact points')
                # from IPython import embed; embed()
                contact_directions_01 = contacts[:,0,:] - contacts[:,1,:]
                all_contact_points = contacts.reshape(-1,3)
                all_finger_diffs = np.maximum(np.linalg.norm(contact_directions_01,axis=1), np.finfo(np.float32).eps)
                all_contact_directions = np.empty((contact_directions_01.shape[0]*2, contact_directions_01.shape[1],))
                all_contact_directions[0::2] = -contact_directions_01 / all_finger_diffs[:,np.newaxis]
                all_contact_directions[1::2] = contact_directions_01 / all_finger_diffs[:,np.newaxis]
                all_contact_suc = np.ones_like(all_contact_points[:,0])
                all_grasp_transform = tfs.reshape(-1,4,4)
                all_approach_directions = all_grasp_transform[:,:3,2]

                pos_idcs = np.where(all_contact_suc>0)[0]
                if len(pos_idcs) == 0:
                    continue
                all_pos_contact_points = all_contact_points[pos_idcs]
                all_pos_finger_diffs = all_finger_diffs[pos_idcs//2]
                all_pos_contact_dirs = all_contact_directions[pos_idcs]
                all_pos_approach_dirs = all_approach_directions[pos_idcs//2]
                all_grasp_transform = all_grasp_transform[pos_idcs//2]

                # Use all positive contacts then mesh_utils with replacement
                if num_pos_contacts > len(all_pos_contact_points)/2:
                    pos_sampled_contact_idcs = np.arange(len(all_pos_contact_points))
                    pos_sampled_contact_idcs_replacement = np.random.choice(np.arange(len(all_pos_contact_points)), num_pos_contacts*2 - len(all_pos_contact_points) , replace=True) 
                    pos_sampled_contact_idcs= np.hstack((pos_sampled_contact_idcs, pos_sampled_contact_idcs_replacement))
                else:
                    pos_sampled_contact_idcs = np.random.choice(np.arange(len(all_pos_contact_points)), num_pos_contacts*2, replace=False)
                pos_contact_points.append(all_pos_contact_points[pos_sampled_contact_idcs,:])
                pos_contact_dirs.append(all_pos_contact_dirs[pos_sampled_contact_idcs,:])
                pos_finger_diffs.append(all_pos_finger_diffs[pos_sampled_contact_idcs])
                pos_approach_dirs.append(all_pos_approach_dirs[pos_sampled_contact_idcs])
                pos_grasp_transforms.append(all_grasp_transform[pos_sampled_contact_idcs,:])
            else:
                pos_contact_points.append(np.zeros((num_pos_contacts*2, 3)))
                pos_contact_dirs.append(np.zeros((num_pos_contacts*2, 3)))
                pos_finger_diffs.append(np.zeros((num_pos_contacts*2,)))
                pos_approach_dirs.append(np.zeros((num_pos_contacts*2, 3)))
                pos_grasp_transforms.append(np.zeros((num_pos_contacts*2, 4, 4)))

        batch_pos_contact_points.append(pos_contact_points)
        batch_pos_contact_dirs.append(pos_contact_dirs)
        batch_pos_finger_diffs.append(pos_finger_diffs)
        batch_pos_approach_dirs.append(pos_approach_dirs)
        batch_pos_grasp_transforms.append(pos_grasp_transforms)

    # embed()
    
    scene_idcs = np.arange(0,len(batch_pos_contact_points))
    contact_points = np.concatenate(batch_pos_contact_points, axis=2)
    grasp_poses = np.concatenate(batch_pos_grasp_transforms, axis=3)
    contact_dirs = np.concatenate(batch_pos_contact_dirs, axis=2)
    finger_diffs = np.concatenate(batch_pos_finger_diffs, axis=1)
    contact_approaches =  np.concatenate(batch_pos_approach_dirs, axis=2)
        
    return contact_points, grasp_poses, contact_dirs, contact_approaches, finger_diffs, scene_idcs

def compute_labels(gt_dict, obs_pcds, cam_poses, data_config):
    """
    Project grasp labels defined on meshes onto rendered point cloud from a camera pose via nearest neighbor contacts within a maximum radius. 
    All points without nearby successful grasp contacts are considered negativ contact points.
    Arguments:
        pos_contact_pts_mesh  -- positive contact points on the mesh scene (Mx3)
        obs_pcds -- observed pointclouds in camera reference frame (bxNx3)
        cam_poses -- pose of each camera in world frame (bx4x4)
        pos_contact_dirs  -- respective contact base directions in the mesh scene (Mx3)
        pos_contact_approaches  -- respective contact approach directions in the mesh scene (Mx3)
        pos_finger_diffs  -- respective grasp widths in the mesh scene (Mx1)
        data_config {dict} -- global config
    Returns:
        [dir_labels_pc_cam, offset_labels_pc, grasp_success_labels_pc, approach_labels_pc_cam] -- 
        Per-point contact success labels and per-contact pose labels in rendered point cloud
    """
    pos_contact_pts_mesh = gt_dict['contact_pts'] # B x SG x F x 3
    pos_contact_dirs = gt_dict['base_dirs'] # B x SG x F x 3
    pos_contact_approaches = gt_dict['approach_dirs'] # B x SG x F x 3
    pos_finger_diffs = gt_dict['offsets'] # B x SG x F
    grasp_poses = gt_dict['grasp_poses'] # B x SG x F x 4 x 4

    with torch.no_grad():
        data_config['k']
        radius = data_config['max_radius']
        data_config['filter_z']
        data_config['z_val']
        b, N = pos_contact_pts_mesh.shape[0], obs_pcds.shape[1] #data_config['num_points']
        pos_contact_pts_mesh = pos_contact_pts_mesh.reshape(b, -1, 3)
        dir_labels = []
        approach_labels = []
        width_labels = []
        success_labels = []
        label_idxs = []
        pose_labels = []
        for i, (pcd, cam_pose, gt_pcd, gt_pose, gt_dir, gt_appr, gt_width) in enumerate(zip(obs_pcds, cam_poses, pos_contact_pts_mesh, grasp_poses, \
                                                                                            pos_contact_dirs, pos_contact_approaches, pos_finger_diffs)):
            pose_labels.append(gt_pose)
            # Find K nearest neighbors to each point from labeled contacts
            knn_tree = KDTree(gt_pcd)
            d, indices = knn_tree.query(pcd, distance_upper_bound=radius)
            
            # Create corresponding lists for baseline, approach, width, and binary success
            dirs = np.zeros_like(pcd)
            approaches = np.zeros_like(pcd)
            widths = np.zeros([N, 1])
            idx_array = []
            gt_dir = gt_dir
            gt_appr = gt_appr

            # print(indices.shape)
            # from IPython import embed; embed()
            for pcd_i, label_i in enumerate(indices):
                # print(label_i)
                # print(len(gt_pcd))
                if label_i == len(gt_pcd):
                    continue
                    
                dir_label = gt_dir[label_i]
                appr_label = gt_appr[label_i]
                width_label = gt_width[0, :][label_i]
                idx_array.append([pcd_i, label_i])
                dirs[pcd_i] = dir_label
                approaches[pcd_i] = appr_label
                widths[pcd_i] = width_label

            label_idxs.append(np.array(idx_array))
            success = np.where(widths>0, 1, 0)

            dir_labels.append(dirs)
            approach_labels.append(approaches)
            width_labels.append(widths)
            success_labels.append(success)

        pose_labels = torch.Tensor(np.stack(pose_labels))
        dir_labels = torch.Tensor(np.stack(dir_labels)).float()
        width_labels = torch.Tensor(np.stack(width_labels)).float()
        success_labels = torch.Tensor(np.stack(success_labels)).float()
        approach_labels = torch.Tensor(np.stack(approach_labels)).float()

        return [pose_labels, label_idxs, dir_labels, width_labels, success_labels, approach_labels]


def compute_labels_single(gt_dict, pcd_list, cam_pose, data_config):
    """
    Project grasp labels defined on meshes onto rendered point cloud from a camera pose via nearest neighbor contacts within a maximum radius. 
    All points without nearby successful grasp contacts are considered negativ contact points.
    Arguments:
        gt_dict  -- ground truth dictionary
        obs_pcds -- observed pointclouds in camera reference frame (bxNx3)
        cam_poses -- pose of each camera in world frame (bx4x4)
        data_config {dict} -- global config
    Returns:
        [dir_labels_pc_cam, offset_labels_pc, grasp_success_labels_pc, approach_labels_pc_cam] -- 
        Per-point contact success labels and per-contact pose labels in rendered point cloud
    """
    pos_contact_pts_mesh = gt_dict['contact_pts'] # B x SG x F x 3
    pos_contact_dirs = gt_dict['base_dirs'] # B x SG x F x 3
    pos_contact_approaches = gt_dict['approach_dirs'] # B x SG x F x 3
    pos_finger_diffs = gt_dict['offsets'] # B x SG x F
    grasp_poses = gt_dict['grasp_poses'] # B x SG x F x 4 x 4
    sg, _F, N = pos_contact_pts_mesh.shape[0], pos_contact_pts_mesh.shape[1], pcd_list.shape[1]
    
    data_config['k']
    radius = 0.008 #data_config['max_radius']
    data_config['filter_z']
    data_config['z_val']
    
    with torch.no_grad():
        pos_contact_pts_mesh = pos_contact_pts_mesh.reshape(sg, -1, 3)
        dir_labels = []
        approach_labels = []
        width_labels = []
        success_labels = []
        label_idxs = []
        pose_labels = []

        for i, (pcd, gt_pcd, gt_pose, gt_dir, gt_appr, gt_width) in enumerate(zip(pcd_list, pos_contact_pts_mesh, grasp_poses, \
                                                                                            pos_contact_dirs, pos_contact_approaches, pos_finger_diffs)):
            pose_labels.append(gt_pose)
            # Find K nearest neighbors to each point from labeled contacts
            knn_tree = KDTree(gt_pcd)
            d, indices = knn_tree.query(pcd, distance_upper_bound=radius)

            # Create corresponding lists for baseline, approach, width, and binary success
            dirs = np.zeros_like(pcd)
            approaches = np.zeros_like(pcd)
            widths = np.zeros([N, 1])
            idx_array = []

            for pcd_i, label_i in enumerate(indices):
                if label_i == len(gt_pcd):
                    continue
                dir_label = gt_dir[label_i]
                appr_label = gt_appr[label_i]
                width_label = gt_width[label_i]
                gt_pose[label_i]
                idx_array.append([pcd_i, label_i])
                dirs[pcd_i] = dir_label
                approaches[pcd_i] = appr_label
                widths[pcd_i] = width_label

            label_idxs.append(np.array(idx_array))

            success = np.where(widths>0, 1, 0)

            dir_labels.append(dirs)
            approach_labels.append(approaches)
            width_labels.append(widths)
            success_labels.append(success)

        pose_labels = torch.Tensor(np.stack(pose_labels))
        dir_labels = torch.Tensor(np.stack(dir_labels)).float()
        width_labels = torch.Tensor(np.stack(width_labels)).float()
        success_labels = torch.Tensor(np.stack(success_labels)).float()
        approach_labels = torch.Tensor(np.stack(approach_labels)).float()

        return [pose_labels, label_idxs, dir_labels, approach_labels, width_labels, success_labels]

    
def compute_labels_aux(gt_dict, obs_pcds, cam_poses, data_config):
    """
    Project grasp labels defined on meshes onto rendered point cloud from a camera pose via nearest neighbor contacts within a maximum radius. 
    All points without nearby successful grasp contacts are considered negativ contact points.
    Arguments:
        gt_dict  -- ground truth dictionary
        obs_pcds -- observed pointclouds in camera reference frame (bxNx3)
        cam_poses -- pose of each camera in world frame (bx4x4)
        data_config {dict} -- global config
    Returns:
        [dir_labels_pc_cam, offset_labels_pc, grasp_success_labels_pc, approach_labels_pc_cam] -- 
        Per-point contact success labels and per-contact pose labels in rendered point cloud
    """
    contacts = gt_dict['contact_pts'] # B x SG x F x 3
    bases = gt_dict['base_dirs'] # B x SG x F x 3
    apprs = gt_dict['approach_dirs'] # B x SG x F x 3
    widths = gt_dict['offsets'] # B x SG x F
    poses  = gt_dict['grasp_poses'] # B x SG x F x 4 x 4
    collision_labels = gt_dict['collision_labels'] # B x SG (booleans)
    _b, sg, _F, N = contacts.shape[0], contacts.shape[1], contacts.shape[2], obs_pcds.shape[2] #data_config['num_points']
    
    # collision_labels = torch.cat([c.expand(1, b) for c in collision_labels], dim=0)
    # collision_labels = collision_labels.T
    data_config['k']
    radius = 0.008 #data_config['max_radius']
    data_config['filter_z']
    data_config['z_val']

    full_bases = []
    full_apprs = []
    full_widths = []
    full_s = []
    full_idxs = []
    
    with torch.no_grad():
        for pcd_list, pos_contact_pts_mesh, pos_contact_dirs, pos_contact_approaches, pos_finger_diffs, grasp_poses, colliding, cam_pose in \
            zip(obs_pcds, contacts, bases, apprs, widths, poses, collision_labels, cam_poses): # for each sample in batch
            pos_contact_pts_mesh = pos_contact_pts_mesh.reshape(sg, -1, 3)
            dir_labels = []
            approach_labels = []
            width_labels = []
            success_labels = []
            label_idxs = []
            pose_labels = []
            
            for i, (pcd, gt_pcd, gt_pose, gt_dir, gt_appr, gt_width) in enumerate(zip(pcd_list, pos_contact_pts_mesh, grasp_poses, \
                                                                                                pos_contact_dirs, pos_contact_approaches, pos_finger_diffs)):

                pose_labels.append(gt_pose)
                # Find K nearest neighbors to each point from labeled contacts
                knn_tree = KDTree(gt_pcd)
                d, indices = knn_tree.query(pcd, distance_upper_bound=radius)

                # Create corresponding lists for baseline, approach, width, and binary success
                dirs = np.zeros_like(pcd)
                approaches = np.zeros_like(pcd)
                widths = np.zeros([N, 1])
                idx_array = []

                for pcd_i, label_i in enumerate(indices):
                    # print(label_i)
                    # print(len(gt_pcd))
                    if label_i == len(gt_pcd):
                        continue
                    dir_label = gt_dir[label_i]
                    appr_label = gt_appr[label_i]
                    width_label = gt_width[label_i]
                    gt_pose[label_i]
                    idx_array.append([pcd_i, label_i])
                    dirs[pcd_i] = dir_label
                    approaches[pcd_i] = appr_label
                    widths[pcd_i] = width_label

                label_idxs.append(np.array(idx_array))

                success = np.where(widths>0, 1, 0)

                dir_labels.append(dirs)
                approach_labels.append(approaches)
                width_labels.append(widths)
                success_labels.append(success)

            pose_labels = torch.Tensor(np.stack(pose_labels))
            dir_labels = torch.Tensor(np.stack(dir_labels)).float()
            width_labels = torch.Tensor(np.stack(width_labels)).float()
            success_labels = torch.Tensor(np.stack(success_labels)).float()
            approach_labels = torch.Tensor(np.stack(approach_labels)).float()
            
            full_bases.append(torch.where(colliding, torch.empty(N, 3, sg), dir_labels.permute(1,2,0)).permute(2,0,1))
            full_apprs.append(torch.where(colliding, torch.empty(N, 3, sg), approach_labels.permute(1,2,0)).permute(2,0,1))
            full_widths.append(torch.where(colliding, torch.empty(N, 1, sg), width_labels.permute(1,2,0)).permute(2,0,1))
            full_s.append(torch.where(colliding, torch.empty(N, 1, sg), success_labels.permute(1,2,0)).permute(2,0,1))
            full_idxs.append(label_idxs)

        full_bases = torch.cat([c.unsqueeze(0) for c in full_bases], dim=0)
        full_apprs = torch.cat([c.unsqueeze(0) for c in full_apprs], dim=0)
        full_widths = torch.cat([c.unsqueeze(0) for c in full_widths], dim=0)
        full_s = torch.cat([c.unsqueeze(0) for c in full_s], dim=0)

        return [poses, full_idxs, full_bases, full_apprs, full_widths, full_s, collision_labels]#[pose_labels, label_idxs, dir_labels, width_labels, success_labels, approach_labels]

class PointCloudReader:
    """
    Class to load scenes, render point clouds and augment them during training
    Arguments:
        root_folder {str} -- acronym root folder
        batch_size {int} -- number of rendered point clouds per-batch
    Keyword Arguments:
        raw_num_points {int} -- Number of random/farthest point samples per scene (default: {20000})
        estimate_normals {bool} -- compute normals from rendered point cloud (default: {False})
        caching {bool} -- cache scenes in memory (default: {True})
        use_uniform_quaternions {bool} -- use uniform quaternions for camera sampling (default: {False})
        scene_obj_scales {list} -- object scales in scene (default: {None})
        scene_obj_paths {list} -- object paths in scene (default: {None})
        scene_obj_transforms {np.ndarray} -- object transforms in scene (default: {None})
        num_train_samples {int} -- training scenes (default: {None})
        num_test_samples {int} -- test scenes (default: {None})
        use_farthest_point {bool} -- use farthest point sampling to reduce point cloud dimension (default: {False})
        intrinsics {str} -- intrinsics to for rendering depth maps (default: {None})
        distance_range {tuple} -- distance range from camera to center of table (default: {(0.9,1.3)})
        elevation {tuple} -- elevation range (90 deg is top-down) (default: {(30,150)})
        pc_augm_config {dict} -- point cloud augmentation config (default: {None})
        depth_augm_config {dict} -- depth map augmentation config (default: {None})
    """
    def __init__(
        self,
        root_folder,
        batch_size=1,
        raw_num_points = 20000,
        estimate_normals = False,
        caching=True,
        use_uniform_quaternions=False,
        scene_obj_scales=None,
        scene_obj_paths=None,
        scene_obj_transforms=None,
        num_train_samples=None,
        num_test_samples=None,
        use_farthest_point = False,
        intrinsics = None,
        distance_range = (0.9,1.3),
        elevation = (30,150),
        pc_augm_config = None,
        depth_augm_config = None
    ):
        self._root_folder = root_folder
        self._batch_size = batch_size
        self._raw_num_points = raw_num_points
        self._caching = caching
        self._num_train_samples = num_train_samples
        self._num_test_samples = num_test_samples
        self._estimate_normals = estimate_normals
        self._use_farthest_point = use_farthest_point
        self._scene_obj_scales = scene_obj_scales
        self._scene_obj_paths = scene_obj_paths
        self._scene_obj_transforms = scene_obj_transforms
        self._distance_range = distance_range
        self._pc_augm_config = pc_augm_config
        self._depth_augm_config = depth_augm_config

        self._current_pc = None
        self._cache = {}

        self._renderer = SceneRenderer(caching=True, intrinsics=intrinsics)

        if use_uniform_quaternions:
            quat_path = os.path.join(self._root_folder, 'uniform_quaternions/data2_4608.qua')
            quaternions = [l[:-1].split('\t') for l in open(quat_path, 'r').readlines()]

            quaternions = [[float(t[0]),
                            float(t[1]),
                            float(t[2]),
                            float(t[3])] for t in quaternions]
            quaternions = np.asarray(quaternions)
            quaternions = np.roll(quaternions, 1, axis=1)
            self._all_poses = [tra.quaternion_matrix(q) for q in quaternions]
        else:
            self._cam_orientations = []
            self._elevation = np.array(elevation)/180. 
            for az in np.linspace(0, np.pi * 2, 30):
                for el in np.linspace(self._elevation[0], self._elevation[1], 30):
                    self._cam_orientations.append(tra.euler_matrix(0, -el, az))
            self._coordinate_transform = tra.euler_matrix(np.pi/2, 0, 0).dot(tra.euler_matrix(0, np.pi/2, 0))

    def get_cam_pose(self, cam_orientation):
        """
        Samples camera pose on shell around table center 
        Arguments:
            cam_orientation {np.ndarray} -- 3x3 camera orientation matrix
        Returns:
            [np.ndarray] -- 4x4 homogeneous camera pose
        """
        
        distance = self._distance_range[0] + np.random.rand()*(self._distance_range[1]-self._distance_range[0])

        extrinsics = np.eye(4)
        extrinsics[0, 3] += distance
        extrinsics = cam_orientation.dot(extrinsics)

        cam_pose = extrinsics.dot(self._coordinate_transform)
        # table height
        cam_pose[2,3] += self._renderer._table_dims[2]
        cam_pose[:3,:2]= -cam_pose[:3,:2]
        return cam_pose

    def _augment_pc(self, pc):
        """
        Augments point cloud with jitter and dropout according to config
        Arguments:
            pc {np.ndarray} -- Nx3 point cloud
        Returns:
            np.ndarray -- augmented point cloud
        """
        
        # not used because no artificial occlusion
        if 'occlusion_nclusters' in self._pc_augm_config and self._pc_augm_config['occlusion_nclusters'] > 0:
            pc = self.apply_dropout(pc,
                                    self._pc_augm_config['occlusion_nclusters'], 
                                    self._pc_augm_config['occlusion_dropout_rate'])

        if 'sigma' in self._pc_augm_config and self._pc_augm_config['sigma'] > 0:
            pc = provider.jitter_point_cloud(pc[np.newaxis, :, :], 
                                            sigma=self._pc_augm_config['sigma'], 
                                            clip=self._pc_augm_config['clip'])[0]
        
        
        return pc[:,:3]

    def _augment_depth(self, depth):
        """
        Augments depth map with z-noise and smoothing according to config
        Arguments:
            depth {np.ndarray} -- depth map
        Returns:
            np.ndarray -- augmented depth map
        """

        if 'sigma' in self._depth_augm_config and self._depth_augm_config['sigma'] > 0:
            clip = self._depth_augm_config['clip']
            sigma = self._depth_augm_config['sigma']
            noise = np.clip(sigma*np.random.randn(*depth.shape), -clip, clip)
            depth += noise
        if 'gaussian_kernel' in self._depth_augm_config and self._depth_augm_config['gaussian_kernel'] > 0:
            kernel = self._depth_augm_config['gaussian_kernel']
            depth_copy = depth.copy()
            depth = cv2.GaussianBlur(depth,(kernel,kernel),0)
            depth[depth_copy==0] = depth_copy[depth_copy==0]
                
        return depth

    def apply_dropout(self, pc, occlusion_nclusters, occlusion_dropout_rate):
        """
        Remove occlusion_nclusters farthest points from point cloud with occlusion_dropout_rate probability
        Arguments:
            pc {np.ndarray} -- Nx3 point cloud
            occlusion_nclusters {int} -- noof cluster to remove
            occlusion_dropout_rate {float} -- prob of removal
        Returns:
            [np.ndarray] -- N > Mx3 point cloud
        """
        if occlusion_nclusters == 0 or occlusion_dropout_rate == 0.:
            return pc

        labels = farthest_points(pc, occlusion_nclusters, distance_by_translation_point)

        removed_labels = np.unique(labels)
        removed_labels = removed_labels[np.random.rand(removed_labels.shape[0]) < occlusion_dropout_rate]
        if removed_labels.shape[0] == 0:
            return pc
        mask = np.ones(labels.shape, labels.dtype)
        for l in removed_labels:
            mask = np.logical_and(mask, labels != l)
        return pc[mask]
    
    def get_scene_batch(self, scene_idx=None, return_segmap=False, save=False):
        """
        Render a batch of scene point clouds
        Keyword Arguments:
            scene_idx {int} -- index of the scene (default: {None})
            return_segmap {bool} -- whether to render a segmap of objects (default: {False})
            save {bool} -- Save training/validation data to npz file for later inference (default: {False})
        Returns:
            [batch_data, cam_poses, scene_idx] -- batch of rendered point clouds, camera poses and the scene_idx
        """
        dims = 6 if self._estimate_normals else 3
        batch_data = np.empty((self._batch_size, self._raw_num_points, dims), dtype=np.float32)
        cam_poses = np.empty((self._batch_size, 4, 4), dtype=np.float32)

        if scene_idx is None:
            scene_idx = np.random.randint(0,self._num_train_samples)

        obj_paths = [os.path.join(self._root_folder, p) for p in self._scene_obj_paths[scene_idx]]
        mesh_scales = self._scene_obj_scales[scene_idx]
        obj_trafos = self._scene_obj_transforms[scene_idx]

        self.change_scene(obj_paths, mesh_scales, obj_trafos, visualize=False)

        batch_segmap, batch_obj_pcs = [], []
        for i in range(self._batch_size):            
            # 0.005s
            pc_cam, pc_normals, camera_pose, depth, _ = self.render_random_scene(estimate_normals = self._estimate_normals)

            if return_segmap:
                segmap, _, obj_pcs = self._renderer.render_labels(depth, obj_paths, mesh_scales, render_pc=True)
                batch_obj_pcs.append(obj_pcs)
                batch_segmap.append(segmap)

            batch_data[i,:,0:3] = pc_cam[:,:3]
            if self._estimate_normals:
                batch_data[i,:,3:6] = pc_normals[:,:3]
            cam_poses[i,:,:] = camera_pose
            
        if save:
            K = np.array([[616.36529541,0,310.25881958 ],[0,616.20294189,236.59980774],[0,0,1]])
            data = {'depth':depth, 'K':K, 'camera_pose':camera_pose, 'scene_idx':scene_idx}
            if return_segmap:
                data.update(segmap=segmap)
            np.savez('results/{}_acronym.npz'.format(scene_idx), data)

        if return_segmap:
            return batch_data, cam_poses, scene_idx, batch_segmap, batch_obj_pcs
        else:
            return batch_data, cam_poses, scene_idx

    def render_random_scene(self, estimate_normals=False, camera_pose=None):
        """
        Renders scene depth map, transforms to regularized pointcloud and applies augmentations
        Keyword Arguments:
            estimate_normals {bool} -- calculate and return normals (default: {False})
            camera_pose {[type]} -- camera pose to render the scene from. (default: {None})
        Returns:
            [pc, pc_normals, camera_pose, depth] -- [point cloud, point cloud normals, camera pose, depth]
        """
        if camera_pose is None:
            viewing_index = np.random.randint(0, high=len(self._cam_orientations)) #CHANGE IF OVERFIT
            camera_orientation = self._cam_orientations[viewing_index]
            camera_pose = self.get_cam_pose(camera_orientation)

        in_camera_pose = copy.deepcopy(camera_pose)

        # 0.005 s
        rgb, depth, _, camera_pose = self._renderer.render(in_camera_pose, render_pc=False)
        depth = self._augment_depth(depth)
        
        pc = self._renderer._to_pointcloud(depth)

        in_camera_pose[:3,3]

        pc = reject_z_outliers(pc)

        '''
        # transform point cloud to world frame
        #pc_hom = np.concatenate((pc, np.ones((pc.shape[0], 1))), 1).T
        xr = R.from_euler('x', np.pi, degrees=False)
        x_rot = np.eye(4)
        x_rot[:3, :3] = xr.as_matrix()
        pc = np.dot(x_rot, pc.T)
        pc = np.dot(camera_pose, pc).T
        '''
        
        pc = regularize_pc_point_count(pc, self._raw_num_points, use_farthest_point=self._use_farthest_point)
        pc = self._augment_pc(pc)

        return pc, camera_pose, depth

    def change_object(self, cad_path, cad_scale):
        """
        Change object in pyrender scene
        Arguments:
            cad_path {str} -- path to CAD model
            cad_scale {float} -- scale of CAD model
        """

        self._renderer.change_scene([cad_path], [cad_scale],[np.eye(4)])

    def change_scene(self, obj_paths, obj_scales, obj_transforms, visualize=False):
        """
        Change pyrender scene
        Arguments:
            obj_paths {list[str]} -- path to CAD models in scene
            obj_scales {list[float]} -- scales of CAD models
            obj_transforms {list[np.ndarray]} -- poses of CAD models
        Keyword Arguments:
            visualize {bool} -- whether to update the visualizer as well (default: {False})
        """
        self._renderer.change_scene(obj_paths, obj_scales, obj_transforms)
        if visualize:
            self._visualizer.change_scene(obj_paths, obj_scales, obj_transforms)

    def __del__(self):
        print('********** terminating renderer **************')
