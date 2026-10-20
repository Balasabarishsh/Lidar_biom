import open3d as o3d
import numpy as np
import os
import matplotlib.pyplot as plt

def visualize_all_chunks_height_colored(chunk_dir):
    """Combine and visualize all .ply chunks in Open3D with height-based colors."""
    chunk_files = sorted([f for f in os.listdir(chunk_dir) if f.endswith('.ply')])
    all_points = []

    if not chunk_files:
        print("⚠️ No .ply files found in the directory.")
        return

    print(f"Loading {len(chunk_files)} chunks from {chunk_dir}...")

    for i, chunk_file in enumerate(chunk_files):
        chunk_path = os.path.join(chunk_dir, chunk_file)
        pcd = o3d.io.read_point_cloud(chunk_path)
        points = np.asarray(pcd.points)
        if points.size == 0:
            continue
        all_points.append(points)
        print(f"✅ Loaded chunk {i+1}/{len(chunk_files)}: {points.shape[0]:,} points")

    # Combine all points
    all_points = np.vstack(all_points)
    print(f"\nTotal points combined: {all_points.shape[0]:,}")

    # Normalize Z-values for color mapping
    z_values = all_points[:, 2]
    z_norm = (z_values - z_values.min()) / (z_values.max() - z_values.min())

    # Apply height-based colormap (jet)
    cmap = plt.get_cmap('jet')
    colors = cmap(z_norm)[:, :3]

    # Create combined point cloud
    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(all_points)
    combined_pcd.colors = o3d.utility.Vector3dVector(colors)

    # Add coordinate axes for reference
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

    print("🌈 Visualizing all chunks together...")
    o3d.visualization.draw_geometries([combined_pcd, axis],
                                      window_name="LiDAR Combined View (Height Colored)",
                                      width=1280,
                                      height=720,
                                      point_show_normal=False)

# Usage
visualize_all_chunks_height_colored(r"E:\PointCloud_Tree_Modelling\dataset\Spatial_chunks")
