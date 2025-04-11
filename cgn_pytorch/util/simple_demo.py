#!/usr/bin/env python3
"""
Simple Meshcat Demo - Hierarchical Visualization with Content-Based Caching
"""
import numpy as np
import trimesh
import hashlib
import time
import webbrowser
from collections import defaultdict

# Create a very simple standalone demo
def run_simple_demo():
    print("Starting minimalist meshcat demo...")
    
    # Import meshcat inside function to avoid import errors
    import meshcat
    
    # Create visualizer without specifying port (let it choose automatically)
    print("Starting meshcat server...")
    vis = meshcat.Visualizer()
    
    # Print and open URL
    url = vis.url()
    print(f"MeshCat URL: {url}")
    webbrowser.open(url, new=2)
    
    # Clear scene
    vis.delete()
    
    # Create a dictionary for caching
    mesh_cache = {}
    transform_cache = {}
    scene_hierarchy = {}
    mesh_usage = defaultdict(int)
    
    def content_hash(data):
        """Generate a content hash for data"""
        return hashlib.md5(data).hexdigest()
        
    def add_object(name, mesh, parent=None, transform=None, color=None):
        """Add object with hierarchical transforms and caching"""
        # Hash the mesh data for caching
        mesh_bytes = mesh.vertices.tobytes() + mesh.faces.tobytes()
        mesh_id = content_hash(mesh_bytes)
        
        # Track usage and cache
        mesh_usage[mesh_id] += 1
        if mesh_id not in mesh_cache:
            print(f"Caching new mesh {mesh_id}")
            mesh_cache[mesh_id] = mesh
            
        # Setup hierarchy
        full_path = f"{parent}/{name}" if parent else name
        scene_hierarchy[full_path] = {
            'mesh_id': mesh_id,
            'parent': parent,
            'children': []
        }
        
        if parent and parent in scene_hierarchy:
            scene_hierarchy[parent]['children'].append(full_path)
            
        # Store transform
        if transform is not None:
            transform_cache[full_path] = transform
            
        # Create visualization
        if color is None:
            color = 0x808080
        material = meshcat.geometry.MeshLambertMaterial(color=color, reflectivity=0.5)
        vis[full_path].set_object(
            meshcat.geometry.TriangularMeshGeometry(mesh.vertices, mesh.faces),
            material
        )
        
        # Apply transform
        if transform is not None:
            vis[full_path].set_transform(transform)
            
        return full_path
        
    def update_transform(name, new_transform):
        """Update transform and propagate to children"""
        if name not in scene_hierarchy:
            return
            
        # Update cache
        transform_cache[name] = new_transform
        
        # Apply to visualization
        vis[name].set_transform(new_transform)
        
        # Update children
        update_children(name, new_transform)
        
    def update_children(parent, parent_transform):
        """Recursively update children"""
        for child in scene_hierarchy[parent]['children']:
            child_local = transform_cache.get(child, np.eye(4))
            child_world = parent_transform @ child_local
            vis[child].set_transform(child_world)
            update_children(child, child_world)
    
    # Create scene objects
    print("Building scene...")
    
    # Base platform (blue)
    box = trimesh.creation.box((0.5, 0.5, 0.1))
    base_name = add_object("base", box, transform=np.eye(4), color=0x3080A0)
    
    # Cylinder on top (purple)
    cylinder = trimesh.creation.cylinder(radius=0.1, height=0.3)
    cyl_tf = np.eye(4)
    cyl_tf[2, 3] = 0.2  # Move up
    cyl_name = add_object("cylinder", cylinder, parent="base", transform=cyl_tf, color=0xA03080)
    
    # Sphere on top of cylinder (green)
    sphere = trimesh.creation.icosphere(radius=0.1)
    sphere_tf = np.eye(4)
    sphere_tf[2, 3] = 0.3  # Move up
    sphere_name = add_object("sphere", sphere, parent="cylinder", transform=sphere_tf, color=0x80A030)
    
    # Another sphere to demonstrate caching (orange)
    sphere2_tf = np.eye(4)
    sphere2_tf[0, 3] = 0.5  # Move right
    sphere2_name = add_object("sphere2", sphere, transform=sphere2_tf, color=0xF0A040)
    
    # Show caching results
    print(f"Sphere mesh ID: {scene_hierarchy[sphere_name]['mesh_id']}")
    print(f"Sphere2 mesh ID: {scene_hierarchy[sphere2_name]['mesh_id']}")
    print(f"Identical meshes: {scene_hierarchy[sphere_name]['mesh_id'] == scene_hierarchy[sphere2_name]['mesh_id']}")
    print(f"Mesh usage count: {mesh_usage[scene_hierarchy[sphere_name]['mesh_id']]}")
    print(f"Cache size: {len(mesh_cache)} unique meshes")
    
    # Animation
    print("Running animation... (press Ctrl+C to stop)")
    try:
        for i in range(200):
            # Rotate the base
            angle = i * 0.05
            rot = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
            update_transform("base", rot)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("Animation stopped")
        
    print("Demo complete!")

if __name__ == "__main__":
    run_simple_demo() 