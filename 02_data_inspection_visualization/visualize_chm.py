import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os

def compare_original_vs_normalized(ground_dir, canopy_dir, normalized_dir, chunk_number=1):
    """Visualize original vs normalized canopy points side-by-side"""
    
    # Load point clouds
    ground_file = f"ground_chunk_{chunk_number}.ply"
    canopy_file = f"canopy_chunk_{chunk_number}.ply"
    normalized_file = f"normalized_canopy_chunk_{chunk_number}.ply"
    
    ground_pcd = o3d.io.read_point_cloud(os.path.join(ground_dir, ground_file))
    original_canopy_pcd = o3d.io.read_point_cloud(os.path.join(canopy_dir, canopy_file))
    normalized_canopy_pcd = o3d.io.read_point_cloud(os.path.join(normalized_dir, normalized_file))
    
    # Color coding
    ground_pcd.paint_uniform_color([0.6, 0.4, 0.2])      # Brown for ground
    original_canopy_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # Red for original canopy
    normalized_canopy_pcd.paint_uniform_color([0.0, 1.0, 0.0]) # Green for normalized canopy
    
    # Visualize separately
    print("=== ORIGINAL CANOPY (Red) + GROUND (Brown) ===")
    o3d.visualization.draw_geometries([ground_pcd, original_canopy_pcd], 
                                    window_name="Original Canopy + Ground")
    
    print("=== NORMALIZED CANOPY (Green) ===")
    o3d.visualization.draw_geometries([normalized_canopy_pcd], 
                                    window_name="Normalized Canopy (Height Above Ground)")
    
    # Visualize together for comparison
    print("=== COMPARISON: Original (Red) vs Normalized (Green) ===")
    o3d.visualization.draw_geometries([original_canopy_pcd, normalized_canopy_pcd], 
                                    window_name="Original vs Normalized")

# Usage
compare_original_vs_normalized(
    ground_dir=r"E:\PointCloud_Tree_Modelling\dataset\improved_ground_filtered\ground",
    canopy_dir=r"E:\PointCloud_Tree_Modelling\dataset\improved_ground_filtered\canopy", 
    normalized_dir=r"E:\PointCloud_Tree_Modelling\dataset\normalized\normalized_canopy",
    chunk_number=1
)