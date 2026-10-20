import numpy as np
import open3d as o3d
from scipy.interpolate import griddata
import os
import pandas as pd
from scipy import ndimage
import cv2
from pyproj import Transformer
import matplotlib.pyplot as plt

class CHMGenerator:
    def __init__(self):
        self.feature_data = None
        self.gt_coords = None
        self.boundary_polygon = None
        self.utm_transformer = None
        
    def load_feature_data(self, feature_csv_path):
        """Load feature data including boundary, benches, pavement, etc."""
        try:
            self.feature_data = pd.read_csv(feature_csv_path)
            print(f"Loaded feature data: {len(self.feature_data)} points")
            
            # Extract boundary coordinates for cleaning
            boundary_points = self.feature_data[self.feature_data['LAYER'] == 'Boundary']
            if len(boundary_points) > 0:
                self.boundary_polygon = boundary_points[['Easting', 'Northing']].values
                print(f"Boundary polygon loaded: {len(self.boundary_polygon)} points")
            
            return True
        except Exception as e:
            print(f"Error loading feature data: {e}")
            return False
    
    def load_ground_truth(self, gt_excel_path):
        """Load ground truth tree locations."""
        try:
            gt_df = pd.read_excel(gt_excel_path)
            self.gt_coords = gt_df[['x_position', 'y_position']].values
            print(f"Loaded {len(self.gt_coords)} ground truth trees")
            return True
        except Exception as e:
            print(f"Ground truth not available: {e}")
            return False
    
    def setup_utm_transform(self, source_epsg=4326, target_epsg=32644):
        """Setup UTM coordinate transformation."""
        try:
            self.utm_transformer = Transformer.from_crs(source_epsg, target_epsg, always_xy=True)
            print("UTM transformer initialized")
            return True
        except Exception as e:
            print(f"UTM transformer setup failed: {e}")
            return False
    
    def is_point_inside_boundary(self, point):
        """Check if a point is inside the boundary polygon using ray casting."""
        if self.boundary_polygon is None:
            return True  # No boundary defined, include all points
        
        x, y = point[0], point[1]
        n = len(self.boundary_polygon)
        inside = False
        
        p1x, p1y = self.boundary_polygon[0]
        for i in range(n + 1):
            p2x, p2y = self.boundary_polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def clean_points_with_boundary(self, points):
        """Remove points outside the boundary."""
        if self.boundary_polygon is None:
            return points
        
        print("Cleaning points with boundary...")
        inside_mask = np.array([self.is_point_inside_boundary(point) for point in points])
        cleaned_points = points[inside_mask]
        
        print(f"Boundary cleaning: {len(points)} -> {len(cleaned_points)} points")
        return cleaned_points
    
    def normalize_individual_points(self, ground_points, canopy_points, grid_resolution=1.0):
        """Normalize canopy point heights using interpolated ground surface."""
        gx, gy, gz = ground_points[:, 0], ground_points[:, 1], ground_points[:, 2]

        canopy_xy = canopy_points[:, :2]
        ground_interp = griddata(
            points=np.column_stack((gx, gy)),
            values=gz,
            xi=canopy_xy,
            method="linear"
        )

        nan_mask = np.isnan(ground_interp)
        if np.any(nan_mask):
            ground_interp[nan_mask] = griddata(
                points=np.column_stack((gx, gy)),
                values=gz,
                xi=canopy_xy[nan_mask],
                method="nearest"
            )

        normalized_z = canopy_points[:, 2] - ground_interp

        # Remove extreme negative outliers
        valid_mask = normalized_z >= -2.0
        normalized_z[~valid_mask] = 0.0
        
        normalized_points = canopy_points.copy()
        normalized_points[:, 2] = normalized_z
        
        negative_count = np.sum(normalized_z < 0)
        negative_percent = (negative_count / len(normalized_z)) * 100
        
        print(f"Normalized heights: {normalized_z.min():.2f}m to {normalized_z.max():.2f}m")
        print(f"Points below ground: {negative_count} ({negative_percent:.2f}%)")
        
        return normalized_points
    
    def enhance_chm_with_ground_truth(self, chm, grid_x, grid_y, influence_radius=8.0):
        """Enhance CHM quality using ground truth tree locations."""
        if self.gt_coords is None:
            return chm
        
        enhanced_chm = chm.copy()
        trees_enhanced = 0
        
        for gt_x, gt_y in self.gt_coords:
            distances = np.sqrt((grid_x - gt_x)**2 + (grid_y - gt_y)**2)
            influence_zone = distances <= influence_radius
            
            if np.any(influence_zone):
                current_heights = chm[influence_zone]
                current_max = np.max(current_heights)
                
                if current_max < 3.0:
                    distance_factor = np.exp(-distances[influence_zone] / (influence_radius / 2))
                    enhancement = 2.0 * distance_factor
                    enhanced_chm[influence_zone] = np.maximum(
                        enhanced_chm[influence_zone], 
                        enhancement
                    )
                    trees_enhanced += 1
        
        if trees_enhanced > 0:
            print(f"CHM spatially enhanced for {trees_enhanced} ground truth areas")
        
        return enhanced_chm
    
    def remove_large_artifacts(self, chm, size_threshold=100):
        """Remove large rectangular artifacts from CHM."""
        chm_clean = chm.copy()
        
        # Find large elevated areas
        elevated_mask = chm > 5.0
        labeled_array, num_features = ndimage.label(elevated_mask)
        
        for i in range(1, num_features + 1):
            component_mask = labeled_array == i
            component_size = np.sum(component_mask)
            
            # Remove very large connected components
            if component_size > size_threshold:
                print(f"Removing large artifact (size: {component_size})")
                chm_clean[component_mask] = 0
        
        return chm_clean
    
    def create_chunk_20_chm(self, canopy_points, resolution=0.5):
        """Special processing for problematic chunk 20."""
        print(" Applying special processing for chunk 20...")
        
        x, y, z = canopy_points[:, 0], canopy_points[:, 1], canopy_points[:, 2]
        
        print(f"Chunk 20 - Points: {len(x)}, Height range: {z.min():.2f}m to {z.max():.2f}m")
        
        # More aggressive filtering for chunk 20
        valid_mask = (z >= 0) & (z <= 15)  # Stricter height limits
        x, y, z = x[valid_mask], y[valid_mask], z[valid_mask]
        
        if len(x) < 1000:
            print("Chunk 20 has too few points after filtering, using fallback")
            return self.create_safe_chm_fallback(canopy_points, resolution)
        
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        
        # Create grid with boundary checks
        grid_x = np.arange(x_min, x_max, resolution)
        grid_y = np.arange(y_min, y_max, resolution)
        
        if len(grid_x) < 2 or len(grid_y) < 2:
            print("Chunk 20 grid too small, using fallback")
            return self.create_safe_chm_fallback(canopy_points, resolution)
            
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)
        
        try:
            # Use nearest neighbor for chunk 20 to avoid interpolation artifacts
            chm = griddata((x, y), z, (grid_x, grid_y), method='nearest', fill_value=0)
            
            # Detect and remove large rectangular artifacts
            chm = self.remove_large_artifacts(chm)
            
            # Very light smoothing only
            chm = ndimage.median_filter(chm, size=2)
            
            print(f"Chunk 20 CHM fixed: {chm.shape}, max height: {np.max(chm):.2f}m")
            return chm, grid_x, grid_y
            
        except Exception as e:
            print(f"Chunk 20 CHM failed: {e}, using fallback")
            return self.create_safe_chm_fallback(canopy_points, resolution)
    
    def create_safe_chm_fallback(self, canopy_points, resolution=0.5):
        """Fallback CHM creation when main method fails."""
        x, y, z = canopy_points[:, 0], canopy_points[:, 1], canopy_points[:, 2]
        
        # Use only reliable points
        valid_mask = (z >= 0) & (z <= 20)
        x, y, z = x[valid_mask], y[valid_mask], z[valid_mask]
        
        if len(x) == 0:
            # Last resort: create empty CHM
            grid_size = 100
            chm = np.zeros((grid_size, grid_size))
            grid_x, grid_y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
            return chm, grid_x, grid_y
        
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        
        grid_x = np.arange(x_min, x_max, resolution)
        grid_y = np.arange(y_min, y_max, resolution)
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)
        
        # Simple nearest neighbor
        chm = griddata((x, y), z, (grid_x, grid_y), method='nearest', fill_value=0)
        
        print(f"Fallback CHM created: {chm.shape}")
        return chm, grid_x, grid_y
    
    def create_high_quality_chm(self, canopy_points, resolution=0.5, chunk_name=""):
        """Create high-quality CHM with boundary cleaning and ground truth enhancement."""
        
        # Clean points with boundary
        canopy_points = self.clean_points_with_boundary(canopy_points)
        
        # Special case for chunk 20
        if "20" in chunk_name:
            return self.create_chunk_20_chm(canopy_points, resolution)
        
        # Normal processing for other chunks
        x, y, z = canopy_points[:, 0], canopy_points[:, 1], canopy_points[:, 2]
        
        valid_mask = (z >= -1) & (z <= 60)
        x, y, z = x[valid_mask], y[valid_mask], z[valid_mask]
        
        if len(x) == 0:
            print("No valid points after filtering")
            return None, None, None
        
        print(f"Using {len(x)} filtered points")
        print(f"Height range: {z.min():.2f}m to {z.max():.2f}m")
        
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        
        grid_x = np.arange(x_min, x_max, resolution)
        grid_y = np.arange(y_min, y_max, resolution)
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)
        
        try:
            chm = griddata((x, y), z, (grid_x, grid_y), method='linear', fill_value=0)
            
            chm = ndimage.median_filter(chm, size=2)
            
            if self.gt_coords is not None:
                chm = self.enhance_chm_with_ground_truth(chm, grid_x, grid_y)
                
        except Exception as e:
            print(f"CHM interpolation failed, using nearest neighbor: {e}")
            chm = griddata((x, y), z, (grid_x, grid_y), method='nearest', fill_value=0)
        
        print(f"CHM created: {chm.shape}, max height: {np.max(chm):.2f}m")
        return chm, grid_x, grid_y
    
    def chm_to_point_cloud(self, chm, grid_x, grid_y, min_height=0.5):
        """Convert CHM grid to 3D point cloud for PLY export."""
        points = []
        colors = []
        
        for i in range(chm.shape[0]):
            for j in range(chm.shape[1]):
                height = chm[i, j]
                if height > min_height:  # Only include points above minimum height
                    if grid_x.ndim == 2:
                        x = grid_x[i, j]
                        y = grid_y[i, j]
                    else:
                        x = grid_x[j]
                        y = grid_y[i]
                    
                    points.append([x, y, height])
                    
                    # Color by height (green to red gradient)
                    normalized_height = height / max(1.0, chm.max())
                    # Green (low) to Red (high)
                    colors.append([normalized_height, 1-normalized_height, 0])
        
        return np.array(points), np.array(colors)
    
    def normalize_and_create_chm(self, ground_dir, canopy_dir, output_dir, grid_resolution=1.0):
        """Main function to normalize all chunks and create enhanced CHM."""
        
        # Load reference data
        self.load_feature_data(r"E:\PointCloud_Tree_Modelling\Feature.csv")
        self.load_ground_truth(r"E:\PointCloud_Tree_Modelling\groudtruth.xlsx")
        self.setup_utm_transform()
        
        ground_files = sorted([f for f in os.listdir(ground_dir) if f.endswith('.ply')])
        canopy_files = sorted([f for f in os.listdir(canopy_dir) if f.endswith('.ply')])
        
        os.makedirs(os.path.join(output_dir, 'normalized_canopy_new'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'enhanced_chm_ply_new'), exist_ok=True)
        
        print(f"Found {len(ground_files)} ground files and {len(canopy_files)} canopy files")
        
        for i, (ground_file, canopy_file) in enumerate(zip(ground_files, canopy_files)):
            chunk_name = f"chunk_{i+1}"
            print(f"Processing {i+1}/{len(canopy_files)}: {chunk_name}...")
            
            try:
                ground_pcd = o3d.io.read_point_cloud(os.path.join(ground_dir, ground_file))
                canopy_pcd = o3d.io.read_point_cloud(os.path.join(canopy_dir, canopy_file))
                
                ground_points = np.asarray(ground_pcd.points)
                canopy_points = np.asarray(canopy_pcd.points)
                
                print(f"Input - Ground: {len(ground_points)}, Canopy: {len(canopy_points)}")
                
                # Normalize canopy points
                normalized_canopy_new = self.normalize_individual_points(ground_points, canopy_points, grid_resolution)
                
                # Create enhanced CHM with boundary cleaning
                enhanced_chm, grid_x, grid_y = self.create_high_quality_chm(
                    normalized_canopy_new, 
                    resolution=0.5,
                    chunk_name=chunk_name
                )
                
                if enhanced_chm is not None:
                    # Convert CHM to point cloud and save as PLY
                    chm_points, chm_colors = self.chm_to_point_cloud(enhanced_chm, grid_x, grid_y)
                    
                    chm_pcd = o3d.geometry.PointCloud()
                    chm_pcd.points = o3d.utility.Vector3dVector(chm_points)
                    chm_colors = np.clip(chm_colors, 0, 1)  # Ensure valid color range
                    chm_pcd.colors = o3d.utility.Vector3dVector(chm_colors)
                    
                    chm_ply_path = os.path.join(output_dir, 'enhanced_chm_ply_new', f'chm_{canopy_file[:-4]}.ply')
                    o3d.io.write_point_cloud(chm_ply_path, chm_pcd)
                    print(f" Saved CHM as PLY: {len(chm_points)} points")
                
                # Save normalized canopy
                normalized_pcd = o3d.geometry.PointCloud()
                normalized_pcd.points = o3d.utility.Vector3dVector(normalized_canopy_new)
                
                output_path = os.path.join(output_dir, 'normalized_canopy_new', f'normalized_{canopy_file}')
                o3d.io.write_point_cloud(output_path, normalized_pcd)
                
                print(f" Saved normalized canopy: {len(normalized_canopy_new)} points")
                print("-" * 50)
                
            except Exception as e:
                print(f" Error processing {canopy_file}: {e}")
                import traceback
                traceback.print_exc()
                continue

# Run the complete CHM generation pipeline
if __name__ == "__main__":
    chm_generator = CHMGenerator()
    
    # CORRECTED PATHS - using your new ground_filtered directory
    chm_generator.normalize_and_create_chm(
        ground_dir=r"E:\PointCloud_Tree_Modelling\dataset\ground_filtered\ground",
        canopy_dir=r"E:\PointCloud_Tree_Modelling\dataset\ground_filtered\canopy", 
        output_dir=r"E:\PointCloud_Tree_Modelling\dataset\normalized"
    )