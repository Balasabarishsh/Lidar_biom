import open3d as o3d
import laspy
import numpy as np
import os
def process_spatial_chunks_quantile(input_path, output_dir, target_chunk_size=5000000):
    """
    Process chunks by dividing based on spatial distribution using quantiles
    """
    print("Reading LAS file...")
    las = laspy.read(input_path)
    points = np.vstack((las.x, las.y, las.z)).T
    total_points = len(points)
    
    print(f"Total points: {total_points:,}")
    
    # Calculate required number of chunks
    num_chunks = max(1, total_points // target_chunk_size)
    if total_points % target_chunk_size != 0:
        num_chunks += 1
    
    print(f"Creating {num_chunks} spatial chunks...")
    
    # Sort points by X coordinate for spatial grouping
    print("Sorting points spatially...")
    sorted_indices = np.argsort(points[:, 0])
    
    os.makedirs(output_dir, exist_ok=True)
    total_clean_points = 0
    chunk_files = []
    
    points_per_chunk = total_points // num_chunks
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * points_per_chunk
        if chunk_idx == num_chunks - 1:  # Last chunk gets all remaining points
            end_idx = total_points
        else:
            end_idx = (chunk_idx + 1) * points_per_chunk
        
        chunk_indices = sorted_indices[start_idx:end_idx]
        chunk_points = points[chunk_indices]
        
        print(f"\nProcessing spatial chunk {chunk_idx+1}/{num_chunks}")
        print(f"Points in this spatial region: {len(chunk_points):,}")
        
        # Calculate spatial bounds
        chunk_min = np.min(chunk_points, axis=0)
        chunk_max = np.max(chunk_points, axis=0)
        print(f"Spatial bounds: X[{chunk_min[0]:.2f}-{chunk_max[0]:.2f}], "
              f"Y[{chunk_min[1]:.2f}-{chunk_max[1]:.2f}]")
        
        # Process chunk
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(chunk_points)
        
        print("Removing statistical outliers...")
        pcd_clean, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        clean_points = np.asarray(pcd_clean.points)
        
        # Save chunk
        chunk_file = os.path.join(output_dir, f"spatial_chunk_{chunk_idx+1}.ply")
        o3d.io.write_point_cloud(chunk_file, pcd_clean)
        chunk_files.append(chunk_file)
        
        total_clean_points += len(clean_points)
        print(f"Spatial chunk {chunk_idx+1}: {len(clean_points):,} clean points -> {chunk_file}")
        
        # Clean up memory
        del pcd, pcd_clean, chunk_points, clean_points
    
    print(f"\nTotal clean points: {total_clean_points:,}")
    print(f"Chunk files saved in: {output_dir}")
    return chunk_files, total_clean_points
def process_spatial_chunks_simple(input_path, output_dir, chunk_size=5000000):
    """
    Simple sequential chunks but with spatial information
    """
    
    print("Reading LAS file...")
    las = laspy.read(input_path)
    total_points = len(las.points)
    num_chunks = (total_points + chunk_size - 1) // chunk_size
    
    print(f"Processing {total_points:,} points in {num_chunks} chunks")
    
    os.makedirs(output_dir, exist_ok=True)
    total_clean_points = 0
    chunk_files = []
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, total_points)
        
        print(f"\nProcessing chunk {chunk_idx+1}/{num_chunks}")
        
        # Read chunk
        chunk_points = las.points[start_idx:end_idx]
        points = np.vstack((chunk_points.x, chunk_points.y, chunk_points.z)).T
        
        # Calculate spatial info
        chunk_min = np.min(points, axis=0)
        chunk_max = np.max(points, axis=0)
        print(f"Spatial bounds: X[{chunk_min[0]:.2f}-{chunk_max[0]:.2f}], "
              f"Y[{chunk_min[1]:.2f}-{chunk_max[1]:.2f}]")    
        # Process
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        pcd_clean, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        clean_points = np.asarray(pcd_clean.points)
        
        # Save
        chunk_file = os.path.join(output_dir, f"chunk_{chunk_idx+1}.ply")
        o3d.io.write_point_cloud(chunk_file, pcd_clean)
        chunk_files.append(chunk_file)
        
        total_clean_points += len(clean_points)
        print(f"Chunk {chunk_idx+1}: {len(clean_points):,} clean points")
        
        del pcd, pcd_clean, points, clean_points
    print(f"\nTotal clean points: {total_clean_points:,}")
    return chunk_files, total_clean_points

# MAIN EXECUTION - CHOOSE ONE METHOD
if __name__ == "__main__":
    input_file = r"E:\PointCloud_Tree_Modelling\dataset\ktp_tranformed.las"
    output_directory = r"E:\PointCloud_Tree_Modelling\dataset\spatial_chunks"
    
    print("Starting spatial chunk processing...")
    
    # METHOD 1: Quantile-based spatial chunks (Recommended)
    chunk_files, total_points = process_spatial_chunks_quantile(
        input_file,
        output_directory
    )
    print(f"\nProcessing completed!")
    print(f"Generated {len(chunk_files)} chunk files")
    print(f"Total clean points: {total_points:,}")