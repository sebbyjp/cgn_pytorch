#!/usr/bin/env python3
"""
Advanced Meshcat Visualizer for RGBD Data, Estimations, and Collisions

Features:
- Hierarchical scene graph visualization
- Content-based mesh caching (Merkle-like)
- Coordinate frame visualization
- Point cloud display
- Handling multiple shape/pose estimations
- Collision detection and visualization
"""
import numpy as np
import trimesh
import trimesh.transformations as tf # Import the submodule explicitly
import hashlib
import time
import webbrowser
import threading
from collections import defaultdict

# Try importing meshcat safely
try:
    import meshcat
    import meshcat.geometry as g
    from meshcat.transformations import translation_matrix, rotation_matrix # Use meshcat's transformations
except ImportError:
    print("Error: meshcat not installed. Please install it: pip install meshcat")
    exit()

# Try importing FCL for collision detection - make it optional
try:
    import fcl
    FCL_AVAILABLE = True
    print("Found python-fcl library. Collision checking enabled.")
except ImportError:

    FCL_AVAILABLE = False

    print("Warning: python-fcl not found. Collision checking will be disabled.")
    print("Install python-fcl for collision features: pip install python-fcl")

# --- Configuration ---
DEFAULT_COLORS = {
    "world_frame": 0xAAAAAA,
    "camera_frame": 0x00FF00,
    "object_frame": 0x0000FF,
    "ground_truth": 0x00CC00,
    "estimation": 0xFF8800,
    "collision": 0xFF0000,
    "point_cloud": 0xCCCCCC,
    "kalman_measurement": 0xFF00FF, # Magenta for noisy measurement
    "kalman_filtered": 0x00FFFF,   # Cyan for filtered estimate
}
FRAME_SCALE = 0.1  # Size of visualized coordinate frames
RAYCAST_SAMPLES = 500 # Number of rays for self-supervision check
KALMAN_NOISE_LEVEL = 0.02 # Std dev for simulated measurement noise
KALMAN_FILTER_GAIN = 0.1 # Simple gain for blending prediction and measurement

# --- Global State ---
vis = None  # Global visualizer instance
mesh_cache = {}
transform_cache = {}
scene_hierarchy = {}
mesh_usage = defaultdict(int)
collision_manager = None

# Kalman Filter State (Example for one object)
kalman_state = {
    'filtered_pose': np.eye(4), # Store the current filtered pose
    'object_path': None         # Path of the object being filtered
}

# --- Helper Functions ---

def get_visualizer():
    """Initializes or returns the global meshcat visualizer."""
    global vis
    if vis is None:
        print("Initializing Meshcat Visualizer...")
        vis = meshcat.Visualizer()
        url = vis.url()
        print(f"MeshCat URL: {url}")
        
        # Open browser in a separate thread
        def open_browser():
            time.sleep(1.0)
            print(f"Opening browser at {url}")
            webbrowser.open(url, new=2)
        threading.Thread(target=open_browser, daemon=True).start()
        
        # Clear any previous state
        vis.delete() 
    return vis

def content_hash(data_bytes):
    """Generate a content hash for mesh data bytes."""
    return hashlib.md5(data_bytes).hexdigest()

def draw_frame(name, transform, scale=FRAME_SCALE, opacity=1.0):
    """Draws a coordinate frame (axes) at the given transform."""
    v = get_visualizer()
    v[name].set_object(g.triad(scale))
    v[name].set_transform(transform)

def draw_frame_experimental(name, transform, scale=FRAME_SCALE, opacity=1.0):
    """Draws a coordinate frame (3D arrows) at the given transform using simple shapes.
    
    Args:
        name (str): The path name in the meshcat hierarchy.
        transform (np.ndarray): The 4x4 transformation matrix for the frame.
        scale (float): The overall length of the arrow shafts.
        opacity (float): Opacity for the materials.
    """
    v = get_visualizer()
    # Ensure the parent path exists and clear any previous geometry
    v[name].delete() 
    
    # Define dimensions based on scale
    shaft_length = scale
    shaft_radius = scale * 0.03
    head_height = scale * 0.2 # Keep head proportional
    head_radius = scale * 0.08 # Keep head proportional

    # Colors
    color_x = 0xFF0000  # Red
    color_y = 0x00FF00  # Green
    color_z = 0x0000FF  # Blue

    # Materials
    mat_x = g.MeshPhongMaterial(color=color_x, transparent=(opacity < 1.0), opacity=opacity)
    mat_y = g.MeshPhongMaterial(color=color_y, transparent=(opacity < 1.0), opacity=opacity)
    mat_z = g.MeshPhongMaterial(color=color_z, transparent=(opacity < 1.0), opacity=opacity)

    # Create reusable geometries
    # Cone points along +Z by default in trimesh, base is at Z=0
    cone_mesh = trimesh.creation.cone(radius=head_radius, height=head_height)
    cone_geom = g.TriangularMeshGeometry(cone_mesh.vertices, cone_mesh.faces)
    # Cylinder is along Z by default in meshcat, centered at origin
    cylinder_geom = g.Cylinder(height=shaft_length, radius=shaft_radius)

    # --- X Axis (Red) --- 
    # Shaft: Default cylinder is along Z. Rotate Z->X. Translate center to (L/2, 0, 0).
    tf_shaft_x = translation_matrix([shaft_length / 2, 0, 0]) @ rotation_matrix(np.pi/2, [0, 1, 0])
    v[name]["x_shaft"].set_object(cylinder_geom, mat_x)
    v[name]["x_shaft"].set_transform(tf_shaft_x)
    # Head: Default cone is along Z. Rotate Z->X. Translate base to (L, 0, 0).
    tf_head_x = translation_matrix([shaft_length, 0, 0]) @ rotation_matrix(np.pi/2, [0, 1, 0])
    v[name]["x_head"].set_object(cone_geom, mat_x)
    v[name]["x_head"].set_transform(tf_head_x)

    # --- Y Axis (Green) ---
    # Shaft: Rotate Z->Y. Translate center to (0, L/2, 0).
    tf_shaft_y = translation_matrix([0, shaft_length / 2, 0]) @ rotation_matrix(-np.pi/2, [1, 0, 0])
    v[name]["y_shaft"].set_object(cylinder_geom, mat_y)
    v[name]["y_shaft"].set_transform(tf_shaft_y)
    # Head: Rotate Z->Y. Translate base to (0, L, 0).
    tf_head_y = translation_matrix([0, shaft_length, 0]) @ rotation_matrix(-np.pi/2, [1, 0, 0])
    v[name]["y_head"].set_object(cone_geom, mat_y)
    v[name]["y_head"].set_transform(tf_head_y)

    # --- Z Axis (Blue) --- 
    # Shaft: No rotation. Translate center to (0, 0, L/2).
    tf_shaft_z = translation_matrix([0, 0, shaft_length / 2])
    v[name]["z_shaft"].set_object(cylinder_geom, mat_z)
    v[name]["z_shaft"].set_transform(tf_shaft_z)
    # Head: No rotation. Translate base to (0, 0, L).
    tf_head_z = translation_matrix([0, 0, shaft_length])
    v[name]["z_head"].set_object(cone_geom, mat_z)
    v[name]["z_head"].set_transform(tf_head_z)

    # Apply the main transform to the parent group node containing all axis parts
    v[name].set_transform(transform)

def add_mesh_to_scene(name, mesh, parent_path=None, transform=np.eye(4), color=None, opacity=1.0, is_estimation=False, add_to_collision=True):
    """Adds a mesh object to the scene graph with caching and hierarchy."""
    global FCL_AVAILABLE, collision_manager # Declare intention to modify global variables
    v = get_visualizer()

    # --- Caching ---
    mesh_bytes = mesh.vertices.tobytes() + mesh.faces.tobytes()
    mesh_id = content_hash(mesh_bytes)
    mesh_usage[mesh_id] += 1
    if mesh_id not in mesh_cache:
        print(f"Caching new mesh {mesh_id} for {name}")
        mesh_cache[mesh_id] = mesh
    
    # --- Hierarchy ---
    full_path = f"{parent_path}/{name}" if parent_path else name
    scene_hierarchy[full_path] = {
        'mesh_id': mesh_id,
        'parent': parent_path,
        'children': [],
        'original_color': color if color is not None else (DEFAULT_COLORS["estimation"] if is_estimation else DEFAULT_COLORS["ground_truth"]),
        'original_opacity': opacity,
        'colliding': False,
        'in_collision_manager': False
    }
    if parent_path and parent_path in scene_hierarchy:
        scene_hierarchy[parent_path]['children'].append(full_path)

    # --- Visualization ---
    final_color = scene_hierarchy[full_path]['original_color']
    material = g.MeshLambertMaterial(color=final_color, reflectivity=0.5, transparent=(opacity < 1.0), opacity=opacity)
    
    # Use cached mesh for geometry
    cached_mesh = mesh_cache[mesh_id]
    v[full_path].set_object(
        g.TriangularMeshGeometry(cached_mesh.vertices, cached_mesh.faces),
        material
    )
    
    # Apply initial transform
    update_transform(full_path, transform) 
    
    # --- Collision (Optional based on FCL availability) ---
    if FCL_AVAILABLE and add_to_collision:
        if collision_manager is None:
            try:
                collision_manager = trimesh.collision.CollisionManager()
            except ValueError as e:
                 print(f"Disabling collision checking: Failed to initialize CollisionManager ({e})")
                 FCL_AVAILABLE = False
                 collision_manager = None # Ensure it's None
            except Exception as e_other:
                 print(f"Unexpected error initializing CollisionManager: {e_other}")
                 FCL_AVAILABLE = False 
                 collision_manager = None

        if collision_manager is not None: # Check if initialization succeeded
            try:
                # Use the initial transform provided
                collision_manager.add_object(name=full_path, mesh=cached_mesh, transform=transform)
                scene_hierarchy[full_path]['in_collision_manager'] = True
                print(f"Added {full_path} to collision manager.")
            except Exception as e:
                print(f"Warning: Could not add {full_path} to collision manager: {e}")
            
    return full_path

def add_point_cloud(name, points, colors=None, size=0.005, transform=np.eye(4)):
    """Adds a point cloud to the visualization."""
    v = get_visualizer()
    
    # Ensure points are Nx3
    if points.shape[1] != 3:
        points = points.T
    if points.shape[1] != 3:
         raise ValueError(f"Point cloud must be Nx3, got {points.shape}")

    # Handle colors (must be Nx3, 0-255)
    if colors is None:
        final_colors = np.tile(np.array([DEFAULT_COLORS["point_cloud"] >> 16 & 0xFF, 
                                         DEFAULT_COLORS["point_cloud"] >> 8 & 0xFF, 
                                         DEFAULT_COLORS["point_cloud"] & 0xFF], dtype=np.uint8), 
                               (points.shape[0], 1))
    elif isinstance(colors, (tuple, list)) and len(colors) == 3: # Single RGB tuple (0-1 float)
        final_colors = np.tile(np.array(np.array(colors) * 255, dtype=np.uint8), (points.shape[0], 1))
    elif isinstance(colors, np.ndarray) and colors.shape == (3,): # Single color np array
        final_colors = np.tile(np.array(colors * 255, dtype=np.uint8), (points.shape[0], 1))
    elif isinstance(colors, np.ndarray) and colors.shape == points.shape:
        final_colors = (colors * 255).astype(np.uint8)
    else:
        print(f"Warning: Point cloud color shape mismatch ({getattr(colors, 'shape', 'N/A')} vs {points.shape}), using default.")
        final_colors = np.tile(np.array([DEFAULT_COLORS["point_cloud"] >> 16 & 0xFF, 
                                         DEFAULT_COLORS["point_cloud"] >> 8 & 0xFF, 
                                         DEFAULT_COLORS["point_cloud"] & 0xFF], dtype=np.uint8), 
                               (points.shape[0], 1))

    # Material type might mismatch linter, but is correct for Points
    v[name].set_object(
        g.Points(
            g.PointsGeometry(points.T, color=final_colors.T), # Meshcat expects 3xN
            material=g.PointsMaterial(size=size) # type: ignore
    ))
    v[name].set_transform(transform)
    
def update_transform(full_path, new_transform):
    """Updates the transform of an object and its children visually and in collision manager."""
    global collision_manager
    v = get_visualizer()

    if full_path not in scene_hierarchy:
        print(f"Warning: Cannot update transform for non-existent path {full_path}")
        return

    # --- Update Cache and Visuals ---
    transform_cache[full_path] = new_transform
    try:
        v[full_path].set_transform(new_transform)
    except KeyError:
        print(f"Warning: Meshcat object {full_path} not found for transform update.")


    # --- Update Collision Manager (Optional) ---
    if FCL_AVAILABLE and collision_manager is not None and scene_hierarchy[full_path].get('in_collision_manager', False):
        try:
            collision_manager.set_transform(name=full_path, transform=new_transform)
        except KeyError:
             mesh_id = scene_hierarchy[full_path]['mesh_id']
             if mesh_id in mesh_cache:
                 try:
                     if full_path not in collision_manager._objs:
                         collision_manager.add_object(name=full_path, mesh=mesh_cache[mesh_id], transform=new_transform)
                         print(f"Re-added {full_path} to collision manager after transform update.")
                     else:
                         collision_manager.set_transform(name=full_path, transform=new_transform)
                 except Exception as e_add:
                     print(f"Warning: Failed to re-add/update {full_path} in collision manager: {e_add}")
             else:
                 print(f"Warning: Mesh {mesh_id} not in cache, cannot update collision for {full_path}")
        except Exception as e_set:
             print(f"Warning: Error setting transform in collision manager for {full_path}: {e_set}")


    # --- Update Children (Recursive) ---
    _update_children_transforms(full_path, new_transform)

def _update_children_transforms(parent_path, parent_world_transform):
    """Helper to recursively update child transforms."""
    if parent_path not in scene_hierarchy:
        return
        
    for child_path in scene_hierarchy[parent_path]['children']:
        child_local_transform = transform_cache.get(child_path, np.eye(4))
        child_world_transform = parent_world_transform @ child_local_transform
        update_transform(child_path, child_world_transform) # Recursively call update_transform

def check_and_visualize_collisions():
    """Checks for collisions and updates object colors accordingly."""
    global collision_manager
    v = get_visualizer()
    # Only run if FCL is available and manager initialized
    if not FCL_AVAILABLE or collision_manager is None:
        return 

    colliding_paths_currently = set()
    try:
        # Check if manager has enough objects to perform checks
        if hasattr(collision_manager, '_objs') and len(collision_manager._objs) > 1: 
            # Unpack defensively, assuming standard return is (bool, set)
            collision_result = collision_manager.in_collision_internal(return_names=True)
            if isinstance(collision_result, tuple) and len(collision_result) >= 2 and isinstance(collision_result[0], bool) and isinstance(collision_result[1], set):
                is_collision, names = collision_result[0], collision_result[1]
                if is_collision:
                    colliding_paths_currently = names
            else:
                print(f"Warning: Unexpected return type from in_collision_internal: {type(collision_result)}")
    except Exception as e:
        print(f"Error during collision detection: {e}")
        return # Skip visual updates if detection failed

    # Update visuals based on collision state changes
    for path, data in scene_hierarchy.items():
        # Only update visuals for objects potentially involved in collisions
        if not data.get('in_collision_manager', False):
            continue 
            
        is_currently_colliding = path in colliding_paths_currently
        was_previously_colliding = data['colliding']

        if is_currently_colliding != was_previously_colliding:
            data['colliding'] = is_currently_colliding
            try:
                if is_currently_colliding:
                    # Set collision appearance
                    new_color = DEFAULT_COLORS["collision"]
                    new_opacity = 1.0
                    # print(f"Setting collision material for {path}") # Debug
                else:
                    # Reset to original appearance
                    new_color = data['original_color']
                    new_opacity = data['original_opacity']
                    # print(f"Resetting material for {path}") # Debug
                    
                # Update material properties using set_property for individual fields
                v[path].set_property("material.color", new_color)
                v[path].set_property("material.opacity", new_opacity)
                v[path].set_property("material.transparent", new_opacity < 1.0)
                
            except Exception as e:
                print(f"Warning: Could not update material for {path}: {e}")

# --- Self-Supervision Function ---

def calculate_ray_consistency_error(mesh_path1, mesh_path2, num_rays=RAYCAST_SAMPLES):
    """Calculates geometric consistency error between two meshes using ray casting."""
    if mesh_path1 not in scene_hierarchy or mesh_path2 not in scene_hierarchy:
        return np.inf # Indicate error
        
    mesh1_id = scene_hierarchy[mesh_path1]['mesh_id']
    mesh2_id = scene_hierarchy[mesh_path2]['mesh_id']

    if mesh1_id not in mesh_cache or mesh2_id not in mesh_cache:
        print(f"Warning: One or both meshes not in cache for ray casting.")
        return np.inf

    mesh1 = mesh_cache[mesh1_id]
    mesh2 = mesh_cache[mesh2_id]
    
    # Get current world transforms
    tf1 = transform_cache.get(mesh_path1, np.eye(4))
    tf2 = transform_cache.get(mesh_path2, np.eye(4))
    
    # Sample points on mesh1's local geometry
    # (Using mesh1 as the reference for sampling)
    origins_local = mesh1.sample(num_rays)
    # Transform sample origins to world frame using mesh1's transform
    origins_world = trimesh.transform_points(origins_local, tf1)
    # Define ray directions in world frame (random directions)
    directions_world = np.random.randn(num_rays, 3)
    directions_world = trimesh.unitize(directions_world)

    # Intersect rays with both meshes. We need hit points in world coordinates.
    # Option 1: Transform meshes to world frame and intersect world rays.
    # Option 2: Transform world rays into each mesh's local frame, intersect,
    #           then transform local hits back to world frame.
    # Option 2 is generally preferred as transforming rays is cheaper than copying/transforming meshes.

    try:
        # Transform world rays into the local frame of mesh1
        origins_mesh1_frame = trimesh.transform_points(origins_world, np.linalg.inv(tf1))
        directions_mesh1_frame = trimesh.transform_points(directions_world, np.linalg.inv(tf1), translate=False)

        # Transform world rays into the local frame of mesh2
        origins_mesh2_frame = trimesh.transform_points(origins_world, np.linalg.inv(tf2))
        directions_mesh2_frame = trimesh.transform_points(directions_world, np.linalg.inv(tf2), translate=False)

        # Intersect rays with meshes in their respective local frames
        hits1_local, index_ray1, face_indices1 = mesh1.ray.intersects_location(
            ray_origins=origins_mesh1_frame, ray_directions=directions_mesh1_frame, multiple_hits=False)
        hits2_local, index_ray2, face_indices2 = mesh2.ray.intersects_location(
            ray_origins=origins_mesh2_frame, ray_directions=directions_mesh2_frame, multiple_hits=False)

        # If no rays hit one of the meshes, consistency is undefined/infinite
        if len(index_ray1) == 0 or len(index_ray2) == 0:
            print(f"Warning: Rays hit 0 times on one mesh ({len(index_ray1)} vs {len(index_ray2)}) for {mesh_path1}/{mesh_path2}")
            return np.inf

        # Create dictionaries mapping original ray index to LOCAL hit location
        hits1_dict = dict(zip(index_ray1, hits1_local))
        hits2_dict = dict(zip(index_ray2, hits2_local))

        # Find indices of rays that hit *both* meshes
        # Use sets for efficient intersection
        valid_indices_set = set(index_ray1).intersection(index_ray2)
        if not valid_indices_set:
            # print(f"Warning: No common ray hits between {mesh_path1} and {mesh_path2}")
            return np.inf # Or a large value, indicates no overlap in hits

        valid_indices = list(valid_indices_set)

        # Extract the corresponding LOCAL hit locations for valid rays
        final_hits1_local = np.array([hits1_dict[i] for i in valid_indices])
        final_hits2_local = np.array([hits2_dict[i] for i in valid_indices])

        # Transform LOCAL hit locations to WORLD coordinates
        final_hits1_world = trimesh.transform_points(final_hits1_local, tf1)
        final_hits2_world = trimesh.transform_points(final_hits2_local, tf2)

        # Calculate distances between WORLD hit locations
        distances = np.linalg.norm(final_hits1_world - final_hits2_world, axis=1)
        avg_error = np.mean(distances)
        return avg_error

    except ImportError:
        print("Error during ray intersection: RTree dependency not found. Install 'rtree'.")
        return np.inf
    except Exception as e:
        print(f"Error during ray intersection: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
        return np.inf

# --- Kalman Filter Function (Simplified) ---

def simple_kalman_update_pose(object_path, ground_truth_pose, noise_level=KALMAN_NOISE_LEVEL, gain=KALMAN_FILTER_GAIN):
    """Performs a very simplified Kalman-like update for visualization."""
    global kalman_state
    v = get_visualizer()

    if object_path not in scene_hierarchy:
        print(f"Warning: Cannot apply Kalman filter to non-existent object: {object_path}")
        return
        
    kalman_state['object_path'] = object_path # Track which object we are filtering

    # 1. Prediction (Simplistic: assume current filtered pose is prediction)
    predicted_pose = kalman_state['filtered_pose']
    
    # 2. Simulate Measurement (Noisy ground truth)
    noise_translation = np.random.normal(scale=noise_level, size=3)
    noise_rotation_angle = np.random.normal(scale=noise_level * 0.5) # Smaller rotation noise
    noise_rotation_axis = trimesh.unitize(np.random.randn(3))
    noise_tf = translation_matrix(noise_translation) @ rotation_matrix(noise_rotation_angle, noise_rotation_axis)
    measurement_pose = ground_truth_pose @ noise_tf
    
    # 3. Update (Simple blending - NOT a proper Kalman filter)
    pred_trans = predicted_pose[:3, 3]
    meas_trans = measurement_pose[:3, 3]
    filt_trans = (1 - gain) * pred_trans + gain * meas_trans
    
    pred_quat = tf.quaternion_from_matrix(predicted_pose)
    meas_quat = tf.quaternion_from_matrix(measurement_pose)
    # Use slerp for quaternion interpolation - suppress potential type error
    filt_quat = tf.quaternion_slerp(pred_quat, meas_quat, gain) # Use tf alias, Correct function name
    # Convert back to rotation matrix - suppress potential type error
    filt_rot_mat = tf.quaternion_matrix(filt_quat)[:3, :3] # Use tf alias
    
    # Combine into new filtered pose
    filtered_pose = np.eye(4)
    filtered_pose[:3, :3] = filt_rot_mat
    filtered_pose[:3, 3] = filt_trans
    
    kalman_state['filtered_pose'] = filtered_pose
    
    # --- Visualize the components ---
    mesh_id = scene_hierarchy[object_path]['mesh_id']
    if mesh_id in mesh_cache:
        mesh = mesh_cache[mesh_id]
        
        # Measurement (magenta, transparent)
        meas_path = "kalman/measurement/" + object_path.split('/')[-1]
        meas_material = g.MeshLambertMaterial(color=DEFAULT_COLORS["kalman_measurement"], reflectivity=0.1, transparent=True, opacity=0.6)
        v[meas_path].set_object(g.TriangularMeshGeometry(mesh.vertices, mesh.faces), meas_material)
        v[meas_path].set_transform(measurement_pose)
        
        # Filtered Estimate (cyan, less transparent)
        filt_path = "kalman/filtered/" + object_path.split('/')[-1]
        filt_material = g.MeshLambertMaterial(color=DEFAULT_COLORS["kalman_filtered"], reflectivity=0.3, transparent=True, opacity=0.8)
        v[filt_path].set_object(g.TriangularMeshGeometry(mesh.vertices, mesh.faces), filt_material)
        v[filt_path].set_transform(filtered_pose)
    else:
        print(f"Warning: Mesh not in cache for Kalman visualization: {mesh_id}")


# --- Demo Execution ---

def run_rgbd_demo():
    """Main function to run the advanced visualization demo."""
    global kalman_state # Allow modification
    print("--- Starting Advanced RGBD Visualization Demo --- ")
    v = get_visualizer() # Initializes visualizer and opens browser

    # --- 1. Setup Frames ---
    print("Setting up coordinate frames...")
    world_frame_tf = np.eye(4)
    draw_frame("frames/world", world_frame_tf)

    # Simulate a camera frame
    camera_tf = translation_matrix([0.5, -0.5, 0.5]) @ \
                rotation_matrix(-np.pi/4, [0, 0, 1]) @ \
                rotation_matrix(-np.pi/6, [0, 1, 0])
    draw_frame("frames/camera", camera_tf)
    transform_cache["frames/camera"] = camera_tf # Store for potential parenting

    # --- 2. Load/Simulate Data ---
    print("Loading/Simulating data...")
    # Simulate a point cloud (e.g., from RGBD sensor in camera frame)
    num_points = 2000
    sim_points_cam_frame = (np.random.rand(num_points, 3) - 0.5) * np.array([1.0, 1.5, 0.3]) # Flat-ish cloud
    sim_points_world_frame = trimesh.transform_points(sim_points_cam_frame, camera_tf)
    add_point_cloud("data/point_cloud", sim_points_world_frame)
    
    # Simulate ground truth objects
    gt_box = trimesh.creation.box([0.2, 0.2, 0.2])
    gt_box_tf = translation_matrix([-0.3, 0.0, 0.1])
    gt_box_path = add_mesh_to_scene("box", gt_box, parent_path="ground_truth", transform=gt_box_tf, 
                                     color=DEFAULT_COLORS["ground_truth"], add_to_collision=True)
    
    gt_sphere = trimesh.creation.icosphere(radius=0.1)
    gt_sphere_tf = translation_matrix([0.1, 0.2, 0.05])
    gt_sphere_path = add_mesh_to_scene("sphere", gt_sphere, parent_path="ground_truth", transform=gt_sphere_tf, 
                                        color=DEFAULT_COLORS["ground_truth"], add_to_collision=True)

    # --- 3. Simulate Estimations ---
    print("Simulating estimations...")
    # Estimation 1: Slightly wrong shape for the box
    est_box_shape = trimesh.creation.box([0.22, 0.18, 0.21])
    add_mesh_to_scene("box_shape_1", est_box_shape, parent_path="estimations/shape", transform=gt_box_tf, 
                       color=DEFAULT_COLORS["estimation"], opacity=0.6, is_estimation=True, add_to_collision=False) # Don't collide shape estimations

    # Estimation 2: Correct shape, slightly wrong pose for the box
    est_box_pose_tf = gt_box_tf @ translation_matrix([0.05, -0.02, 0]) @ \
                      rotation_matrix(0.1, [0, 0, 1])
    est_box_pose_path = add_mesh_to_scene("box_pose_1", gt_box, parent_path="estimations/pose", transform=est_box_pose_tf, 
                                          color=DEFAULT_COLORS["estimation"], opacity=0.6, is_estimation=True, add_to_collision=True) # Collide pose estimations
    
    # Initialize Kalman state with this estimation
    kalman_state['filtered_pose'] = est_box_pose_tf.copy()
    kalman_state['object_path'] = est_box_pose_path

    # Estimation 3: Another wrong pose for the box (colliding with sphere)
    est_box_pose_tf_collide = translation_matrix([0.05, 0.15, 0.05]) # Move towards sphere
    est_box_pose_collide_path = add_mesh_to_scene("box_pose_2_collide", gt_box, parent_path="estimations/pose", transform=est_box_pose_tf_collide, 
                                                color=DEFAULT_COLORS["estimation"], opacity=0.6, is_estimation=True, add_to_collision=True)

    # Estimation 4: Wrong pose for sphere
    est_sphere_pose_tf = gt_sphere_tf @ translation_matrix([-0.03, 0.04, 0.02])
    est_sphere_pose_path = add_mesh_to_scene("sphere_pose_1", gt_sphere, parent_path="estimations/pose", transform=est_sphere_pose_tf, 
                                             color=DEFAULT_COLORS["estimation"], opacity=0.6, is_estimation=True, add_to_collision=True)

    # Demonstrate caching
    print(f"Box mesh ID (GT): {scene_hierarchy[gt_box_path]['mesh_id']}")
    print(f"Box mesh ID (PoseEst): {scene_hierarchy[est_box_pose_path]['mesh_id']}")
    print(f"Box mesh ID (ShapeEst): {scene_hierarchy['estimations/shape/box_shape_1']['mesh_id']}")
    print(f"Sphere mesh ID (GT): {scene_hierarchy[gt_sphere_path]['mesh_id']}")
    print(f"Total unique meshes in cache: {len(mesh_cache)}")
    
    # --- 4. Initial Collision Check ---
    print("Performing initial collision check...")
    check_and_visualize_collisions()

    # --- 5. Animation/Interaction Loop (Example) ---
    print("Starting animation loop...")
    print("Press Ctrl+C to stop.")
    loop_counter = 0
    trajectory_radius = 0.4
    trajectory_speed = 0.03 # Radians per frame
    original_gt_box_tf = gt_box_tf.copy() # Keep original offset from world
    
    try:
        while True: # Loop indefinitely until Ctrl+C
            loop_counter += 1
            current_angle = loop_counter * trajectory_speed
            
            # --- A. Animate the Ground Truth Box --- 
            # Calculate displacement for circular motion in XY plane
            delta_x = trajectory_radius * np.cos(current_angle)
            delta_y = trajectory_radius * np.sin(current_angle)
            # Apply displacement relative to the original GT pose
            circular_motion_tf = translation_matrix([delta_x, delta_y, 0])
            new_gt_box_tf = original_gt_box_tf @ circular_motion_tf
            update_transform(gt_box_path, new_gt_box_tf)
            
            # --- B. Update Kalman Filter Visualization --- 
            # The measurement (magenta) and filtered (cyan) poses will be updated 
            # relative to the *current* ground truth pose inside this function.
            meas_pose = None
            filt_pose = None
            if kalman_state['object_path']: # Only update if KF is initialized
                # Get the poses *after* the update inside the function
                simple_kalman_update_pose(kalman_state['object_path'], new_gt_box_tf) # Update filter based on *moving* GT box
                meas_path = "kalman/measurement/" + kalman_state['object_path'].split('/')[-1]
                filt_path = "kalman/filtered/" + kalman_state['object_path'].split('/')[-1]
                meas_pose = transform_cache.get(meas_path)
                filt_pose = transform_cache.get(filt_path) # Use the updated filtered pose from kalman_state
                
            # Calculate translation errors if poses are available
            if meas_pose is not None and filt_pose is not None:
                gt_trans = new_gt_box_tf[:3, 3]
                meas_trans = meas_pose[:3, 3]
                filt_trans = filt_pose[:3, 3]
                
                error_meas = np.linalg.norm(gt_trans - meas_trans)
                error_filt = np.linalg.norm(gt_trans - filt_trans)
                
                # Print errors periodically
                if loop_counter % 10 == 0:
                     print(f"[Pose Error] Measurement: {error_meas:.4f}, Filtered: {error_filt:.4f}")

            
            # --- C. Perform Self-Supervision Check (Periodically) ---
            # Note: Comparing GT vs the *static* Est1 pose might become less meaningful as GT moves.
            if loop_counter % 20 == 0: # Check every 20 frames
                consistency_error = calculate_ray_consistency_error(gt_box_path, est_box_pose_path)
                print(f"[Self-Supervision] Ray Consistency Error (GT vs Est1): {consistency_error:.4f}")
                
                # Check if kalman filter visualization exists before comparing
                # This check might fail as kalman viz isn't formally in scene_hierarchy
                kalman_viz_path = "kalman/filtered/" + est_box_pose_path.split('/')[-1] if est_box_pose_path else None
                if kalman_viz_path and kalman_viz_path in transform_cache: # Check transform_cache instead
                    try:
                        # Need to add the kalman mesh to scene_hierarchy or handle differently
                        # For now, let's skip this comparison to avoid errors
                        # consistency_error_kalman = calculate_ray_consistency_error(gt_box_path, kalman_viz_path)
                        # print(f"[Self-Supervision] Ray Consistency Error (GT vs Kalman): {consistency_error_kalman:.4f}")
                        pass # Skip Kalman comparison for now
                    except Exception as e_ray_kalman:
                        print(f"Warning: Error during Kalman ray consistency check: {e_ray_kalman}")
            
            # --- D. Check Collisions --- 
            # Collision checking will now happen between the moving GT box/sphere and the static estimates
            check_and_visualize_collisions()
            
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("Animation stopped by user.")
    except Exception as e:
        print(f"An error occurred during animation: {e}")
        import traceback
        traceback.print_exc()

    print("--- Demo Complete --- ")
    print("You can interact with the scene in the Meshcat browser window.")
    input("Press Enter to exit and shut down server...") 
    
if __name__ == "__main__":
    run_rgbd_demo()

"""
Potential Enhancements:
- Load actual RGBD data (PLY, PCD) instead of simulating.
- Add GUI controls (e.g., using ipywidgets in a notebook) to toggle estimations, step through data.
- Implement more sophisticated collision visualization (e.g., showing contact points).
- Integrate concepts from notes.md more directly (e.g., perceptual hashing for shape matching, plane fitting for scene understanding).
- Add garbage collection for the mesh cache based on mesh_usage.
- Implement a more correct Kalman filter (tracking velocity, uncertainty propagation).
""" 