import open3d as o3d
import laspy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ✅ Use raw string for path (avoid escape issues)
las = laspy.read(r"E:\PointCloud_Tree_Modelling\dataset\ktp_tranformed.las")

# Extract XYZ coordinates
points = np.vstack((las.x, las.y, las.z)).T

# Center points for better visualization
points_centered = points - np.mean(points, axis=0)

# Normalize elevation (Z)
z_values = points_centered[:, 2]
z_min, z_max = np.min(z_values), np.max(z_values)
z_norm = (z_values - z_min) / (z_max - z_min)

# ✅ Updated for Matplotlib 3.7+ compatibility
colormap = plt.get_cmap('jet')   # instead of cm.get_cmap('jet')
colors = colormap(z_norm)[:, :3]  # Extract RGB values

# Create Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_centered)
pcd.colors = o3d.utility.Vector3dVector(colors)

# ✅ Visualize in Open3D
o3d.visualization.draw_geometries(
    [pcd],
    window_name="LiDAR Elevation Visualization",
    width=1280,
    height=720
)

# ✅ Show colorbar legend using matplotlib
fig, ax = plt.subplots(figsize=(6, 1))
norm = plt.Normalize(vmin=z_min, vmax=z_max)
cb = plt.colorbar(
    cm.ScalarMappable(norm=norm, cmap=colormap),
    cax=ax,
    orientation='horizontal'
)
cb.set_label('Elevation (m)')
plt.show()
