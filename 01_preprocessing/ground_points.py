import open3d as o3d
import os
import numpy as np
from scipy.spatial import KDTree

def main():
    """Main function to avoid definition order issues"""   
    def better_ground_filter(points, height_threshold=1.5, window_size=5.0):
        """Better ground filtering that preserves tree trunks"""
        min_coords = np.min(points[:, :2], axis=0)
        max_coords = np.max(points[:, :2], axis=0)       
        x_cells = int(np.ceil((max_coords[0] - min_coords[0]) / window_size))
        y_cells = int(np.ceil((max_coords[1] - min_coords[1]) / window_size))        
        ground_mask = np.zeros(len(points), dtype=bool)        
        for i in range(x_cells):
            for j in range(y_cells):
                x_min = min_coords[0] + i * window_size
                x_max = x_min + window_size
                y_min = min_coords[1] + j * window_size
                y_max = y_min + window_size                
                cell_mask = (points[:, 0] >= x_min) & (points[:, 0] < x_max) & \
                           (points[:, 1] >= y_min) & (points[:, 1] < y_max)               
                cell_points = points[cell_mask]                
                if len(cell_points) > 10:
                    sorted_z = np.sort(cell_points[:, 2])
                    ground_level = np.mean(sorted_z[:max(5, len(cell_points)//10)])
                    
                    temp_ground_mask = cell_points[:, 2] < (ground_level + height_threshold)
                    
                    if np.sum(temp_ground_mask) > 5:
                        ground_candidates = cell_points[temp_ground_mask]
                        tree = KDTree(ground_candidates[:, :2])
                        distances, _ = tree.query(ground_candidates[:, :2], k=4)
                        avg_distances = distances.mean(axis=1)                        
                        trunk_threshold = np.percentile(avg_distances, 30)
                        sparse_ground_mask = avg_distances > trunk_threshold                        
                        final_ground_mask = temp_ground_mask.copy()
                        final_ground_mask[temp_ground_mask] = sparse_ground_mask                        
                        ground_indices = np.where(cell_mask)[0][final_ground_mask]
                        ground_mask[ground_indices] = True       
        return points[ground_mask], points[~ground_mask]   
    
    # Process chunks - UPDATED PATHS
    chunk_dir = r"E:\PointCloud_Tree_Modelling\dataset\spatial_chunks"
    output_dir = r"E:\PointCloud_Tree_Modelling\dataset\ground_filtered"    
    chunk_files = sorted([f for f in os.listdir(chunk_dir) if f.endswith('.ply')])   
    os.makedirs(os.path.join(output_dir, 'ground'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'canopy'), exist_ok=True)    
    for chunk_idx, chunk_file in enumerate(chunk_files):
        print(f"Processing {chunk_file} ({chunk_idx+1}/{len(chunk_files)})")       
        chunk_path = os.path.join(chunk_dir, chunk_file)
        pcd = o3d.io.read_point_cloud(chunk_path)
        points = np.asarray(pcd.points)        
        ground_points, canopy_points = better_ground_filter(points, height_threshold=1.5)        
        print(f"Ground: {len(ground_points):,} | Canopy: {len(canopy_points):,}")
        # Save results
        ground_pcd = o3d.geometry.PointCloud()
        ground_pcd.points = o3d.utility.Vector3dVector(ground_points)
        ground_output = os.path.join(output_dir, 'ground', f'ground_{chunk_file}')
        o3d.io.write_point_cloud(ground_output, ground_pcd)
        
        canopy_pcd = o3d.geometry.PointCloud()
        canopy_pcd.points = o3d.utility.Vector3dVector(canopy_points)
        canopy_output = os.path.join(output_dir, 'canopy', f'canopy_{chunk_file}')
        o3d.io.write_point_cloud(canopy_output, canopy_pcd)

# Run the main function
if __name__ == "__main__":
    main()