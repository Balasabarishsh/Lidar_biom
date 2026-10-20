import laspy
import numpy as np

# Path to your LAS file
las_path = r"E:\PointCloud_Tree_Modelling\dataset\ktp_tranformed.las"

# Read the LAS file
las = laspy.read(las_path)

# Extract X, Y, Z coordinates
x = las.x
y = las.y
z = las.z

# Compute elevation statistics
z_min = np.min(z)
z_max = np.max(z)
z_mean = np.mean(z)
z_std = np.std(z)

print(f" Elevation (Z) details for: {las_path}")
print(f"    Min Elevation : {z_min:.3f} m")
print(f"    Max Elevation : {z_max:.3f} m")
print(f"    Mean Elevation: {z_mean:.3f} m")
print(f"    Std Deviation : {z_std:.3f} m")
print(f"    Elevation Range: {z_max - z_min:.3f} m")
