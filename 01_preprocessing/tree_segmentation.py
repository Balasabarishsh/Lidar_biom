# tree_segmentation
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import cv2
import os
import glob
import open3d as o3d
from scipy.interpolate import griddata
from shapely.geometry import Point, Polygon

def load_exclusion_features(feature_csv_path):
    """
    Load pathway and bench points from Feature.csv for exclusion zones only
    """
    print(f"Loading exclusion features from: {feature_csv_path}")
    
    if not os.path.exists(feature_csv_path):
        print("No feature CSV found - proceeding without exclusion zones")
        return None, None, None
    
    df = pd.read_csv(feature_csv_path)
    print(f"Loaded {len(df)} feature points")
    
    pathway_points = []
    bench_points = []
    boundary_points = []
    
    for _, row in df.iterrows():
        easting = row['Easting']
        northing = row['Northing']  
        layer = str(row['LAYER']).lower() if pd.notna(row['LAYER']) else ''
        name = str(row['Name']).lower() if pd.notna(row['Name']) else ''
        
        feature_type = layer + " " + name
        
        if 'path' in feature_type or 'walk' in feature_type or 'trail' in feature_type:
            pathway_points.append((easting, northing))
        elif 'bench' in feature_type or 'seat' in feature_type:
            bench_points.append((easting, northing))
        elif 'bound' in feature_type or 'fence' in feature_type or 'edge' in feature_type:
            boundary_points.append((easting, northing))
        else:
            pathway_points.append((easting, northing))
    
    print(f"Exclusion features: {len(pathway_points)} pathway, {len(bench_points)} bench, {len(boundary_points)} boundary")
    
    return pathway_points, bench_points, boundary_points

def create_boundary_polygon(boundary_points):
    """
    Create a polygon from boundary points to define study area
    """
    if not boundary_points or len(boundary_points) < 3:
        print("Not enough boundary points to create polygon")
        return None
    
    boundary_array = np.array(boundary_points)
    polygon = Polygon(boundary_array)
    
    print(f"Study area boundary created with {len(boundary_points)} points")
    print(f"Boundary area: {polygon.area:.1f} m²")
    
    return polygon

def apply_boundary_constraint(grid_x, grid_y, boundary_polygon):
    """
    Apply boundary constraint to limit analysis to study area only
    """
    if boundary_polygon is None:
        print("No boundary polygon - using full area")
        return np.ones_like(grid_x, dtype=bool)
    
    boundary_mask = np.zeros_like(grid_x, dtype=bool)
    
    boundary_count = 0
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            point = Point(grid_x[i, j], grid_y[i, j])
            if boundary_polygon.contains(point):
                boundary_mask[i, j] = True
                boundary_count += 1
    
    total_pixels = grid_x.size
    print(f"Boundary constraint: {boundary_count}/{total_pixels} pixels inside study area ({boundary_count/total_pixels*100:.1f}%)")
    
    return boundary_mask

def create_exclusion_mask(grid_x, grid_y, pathway_points, bench_points, boundary_points, exclusion_radius=3.0):
    """
    Create mask that excludes areas near pathways, benches, and boundaries
    """
    mask = np.ones_like(grid_x, dtype=bool)
    
    if not pathway_points and not bench_points and not boundary_points:
        print("No exclusion features found - using full area")
        return mask
    
    all_exclusion_points = []
    
    if pathway_points:
        all_exclusion_points.extend(pathway_points)
    if bench_points:
        all_exclusion_points.extend(bench_points)
    if boundary_points:
        all_exclusion_points.extend(boundary_points)
    
    if all_exclusion_points:
        all_exclusion_points = np.array(all_exclusion_points)
        print(f"Total exclusion points: {len(all_exclusion_points)}")
        
        exclusion_count = 0
        for i in range(grid_x.shape[0]):
            for j in range(grid_x.shape[1]):
                point = np.array([grid_x[i, j], grid_y[i, j]])
                distances = np.linalg.norm(all_exclusion_points - point, axis=1)
                min_distance = np.min(distances)
                
                if min_distance < exclusion_radius:
                    mask[i, j] = False
                    exclusion_count += 1
        
        excluded_pixels = np.sum(~mask)
        total_pixels = mask.size
        print(f"Exclusion mask: {excluded_pixels}/{total_pixels} pixels excluded ({excluded_pixels/total_pixels*100:.1f}%)")
    
    return mask

def load_chm_ply_as_grid(ply_path):
    """
    Load enhanced CHM PLY file and convert back to grid format
    """
    try:
        print(f"Loading CHM PLY: {os.path.basename(ply_path)}")
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points)
        
        if len(points) == 0:
            raise ValueError("No points in CHM point cloud")
        
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        
        x_sorted = np.sort(np.unique(x))
        y_sorted = np.sort(np.unique(y))
        
        if len(x_sorted) > 1 and len(y_sorted) > 1:
            resolution_x = x_sorted[1] - x_sorted[0]
            resolution_y = y_sorted[1] - y_sorted[0]
            resolution = (resolution_x + resolution_y) / 2
        else:
            resolution = 0.5
        
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        
        grid_x = np.linspace(x_min, x_max, int((x_max - x_min) / resolution) + 1)
        grid_y = np.linspace(y_min, y_max, int((y_max - y_min) / resolution) + 1)
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)
        
        chm = griddata((x, y), z, (grid_x, grid_y), method='linear', fill_value=0)
        
        print(f"CHM grid: {chm.shape}, resolution: {resolution:.2f}m, height range: {np.min(chm):.1f}m to {np.max(chm):.1f}m")
        return chm, grid_x, grid_y, resolution, x_min, y_min, x_max, y_max
        
    except Exception as e:
        print(f"Error loading CHM PLY {ply_path}: {e}")
        return None, None, None, None, None, None, None, None

def preprocess_chm_with_exclusion(chm, exclusion_mask, boundary_mask):
    """
    Enhanced preprocessing with exclusion zones and boundary constraints
    """
    chm_clean = chm.copy()
    
    final_mask = exclusion_mask & boundary_mask
    chm_clean[~final_mask] = 0
    
    chm_clean = ndi.median_filter(chm_clean, size=2)
    chm_clean = cv2.GaussianBlur(chm_clean, (3, 3), 0.5)
    
    return chm_clean

def enhanced_tree_top_detection(chm, exclusion_mask, boundary_mask, min_distance=6, min_height=2.5):
    """
    Improved tree top detection that respects exclusion zones and boundaries
    """
    final_mask = exclusion_mask & boundary_mask
    search_mask = final_mask & (chm > min_height)
    
    if np.max(chm) < min_height:
        print(f"No significant trees detected (max height: {np.max(chm):.1f}m)")
        return np.zeros_like(chm, dtype=np.int32), np.array([])
    
    coordinates = peak_local_max(
        chm, 
        min_distance=min_distance,
        threshold_abs=min_height,
        threshold_rel=0.3,
        exclude_border=False
    )
    
    valid_coords = []
    for coord in coordinates:
        y, x = coord[0], coord[1]
        
        if not final_mask[y, x]:
            continue
            
        y_min, y_max = max(0, y-3), min(chm.shape[0], y+4)
        x_min, x_max = max(0, x-3), min(chm.shape[1], x+4)
        local_region = chm[y_min:y_max, x_min:x_max]
        
        if len(local_region) > 0:
            local_mean = np.mean(local_region)
            if chm[y, x] > local_mean + 1.5 and chm[y, x] > min_height:
                valid_coords.append([y, x])
    
    coordinates = np.array(valid_coords)
    
    markers = np.zeros_like(chm, dtype=np.int32)
    for i, (y, x) in enumerate(coordinates):
        markers[y, x] = i + 1
    
    print(f"Tree top detection: {len(coordinates)} valid tree tops found")
    return markers, coordinates

def smart_watershed_segmentation(chm_clean, markers, exclusion_mask, boundary_mask):
    """
    Watershed segmentation that respects exclusion zones and boundaries
    """
    try:
        tree_mask = chm_clean > 1.0
        final_mask = tree_mask & exclusion_mask & boundary_mask
        
        chm_inverted = chm_clean.max() - chm_clean
        
        labels = watershed(chm_inverted, markers, mask=final_mask, watershed_line=True)
        
        return labels
    except Exception as e:
        print(f"Watershed segmentation failed: {e}")
        return None

def calculate_tree_metrics(tree_mask, chm, grid_x, grid_y, resolution):
    """
    Calculate tree metrics
    """
    if np.sum(tree_mask) < 8:
        return None
    
    tree_heights = chm[tree_mask]
    max_height = np.max(tree_heights)
    
    if max_height < 2.0:
        return None
    
    height_threshold = max_height * 0.3
    crown_heights = tree_heights[tree_heights > height_threshold]
    
    if len(crown_heights) == 0:
        return None
        
    mean_height = np.mean(crown_heights)
    
    tree_pixels = np.sum(tree_mask)
    tree_area = tree_pixels * (resolution ** 2)
    
    tree_top_positions = np.argwhere(tree_mask & (chm >= max_height * 0.9))
    if len(tree_top_positions) > 0:
        y_center = int(np.mean(tree_top_positions[:, 0]))
        x_center = int(np.mean(tree_top_positions[:, 1]))
    else:
        tree_top_idx = np.argmax(tree_heights)
        tree_top_pos = np.argwhere(tree_mask)[tree_top_idx]
        y_center, x_center = tree_top_pos
    
    if grid_x.ndim == 2:
        tree_x = grid_x[y_center, x_center]
        tree_y = grid_y[y_center, x_center]
    else:
        tree_x = grid_x[x_center]
        tree_y = grid_y[y_center]
    
    crown_radius = np.sqrt(tree_area / np.pi)
    volume = (np.pi * crown_radius**2 * mean_height) / 3
    
    return {
        'max_height': max_height,
        'mean_height': mean_height,
        'area_m2': tree_area,
        'pixel_count': tree_pixels,
        'volume_approx': volume,
        'x_position': tree_x,
        'y_position': tree_y,
        'crown_radius': crown_radius
    }

def validate_tree_metrics(metrics):
    """
    Validate tree metrics
    """
    if metrics is None:
        return False
    
    h = metrics['max_height']
    a = metrics['area_m2']
    
    height_ok = 2.0 <= h <= 30.0
    area_ok = 2.0 <= a <= 400.0
    
    return height_ok and area_ok

def process_chunk_chm_only(ply_path, chunk_name, output_dir, pathway_points, bench_points, boundary_points):
    """
    Process chunk using CHM data only for tree segmentation
    """
    print(f"Processing {chunk_name} with CHM-based segmentation...")
    
    try:
        chm, grid_x, grid_y, resolution, x_min, y_min, x_max, y_max = load_chm_ply_as_grid(ply_path)
        if chm is None:
            return []
        
        boundary_polygon = create_boundary_polygon(boundary_points)
        boundary_mask = apply_boundary_constraint(grid_x, grid_y, boundary_polygon)
        exclusion_mask = create_exclusion_mask(grid_x, grid_y, pathway_points, bench_points, boundary_points)
        
        chm_clean = preprocess_chm_with_exclusion(chm, exclusion_mask, boundary_mask)
        
        markers, coordinates = enhanced_tree_top_detection(chm_clean, exclusion_mask, boundary_mask)
        
        if len(coordinates) == 0:
            print(f"No valid tree tops found in {chunk_name}")
            return []
        
        labels = smart_watershed_segmentation(chm_clean, markers, exclusion_mask, boundary_mask)
        
        if labels is None:
            return []
        
        tree_metrics = []
        for tree_id in np.unique(labels):
            if tree_id == 0:
                continue
                
            tree_mask = labels == tree_id
            metrics = calculate_tree_metrics(tree_mask, chm_clean, grid_x, grid_y, resolution)
            
            if metrics and validate_tree_metrics(metrics):
                metrics.update({
                    'tree_id': tree_id,
                    'chunk_name': chunk_name
                })
                tree_metrics.append(metrics)
        
        print(f"Success {chunk_name}: {len(tree_metrics)} valid trees found")
        return tree_metrics
        
    except Exception as e:
        print(f"Error in {chunk_name}: {e}")
        import traceback
        traceback.print_exc()
        return []

def run_chm_only_segmentation():
    """
    Run tree segmentation using CHM data only
    """
    input_dir = r"E:\PointCloud_Tree_Modelling\dataset\normalized\enhanced_chm_ply_new"
    output_dir = r"E:\PointCloud_Tree_Modelling\dataset\tree_segmentation"
    feature_csv_path = r"E:\PointCloud_Tree_Modelling\Feature.csv"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load exclusion features only (no DGPS tree points)
    pathway_points, bench_points, boundary_points = load_exclusion_features(feature_csv_path)
    
    ply_files = glob.glob(os.path.join(input_dir, "chm_*.ply"))
    ply_files.sort()
    
    all_trees = []
    
    print("\n" + "="*70)
    print("STARTING TREE SEGMENTATION - CHM ONLY")
    print("="*70)
    
    for i, ply_file in enumerate(ply_files):
        chunk_name = f"chunk_{i+1}"
        
        print(f"\n--- Processing {chunk_name} ---")
        trees = process_chunk_chm_only(
            ply_file, chunk_name, output_dir, 
            pathway_points, bench_points, boundary_points
        )
        all_trees.extend(trees)
    
    # Save results
    if all_trees:
        df = pd.DataFrame(all_trees)
        output_csv = os.path.join(output_dir, "trees_dataset.csv")
        df.to_csv(output_csv, index=False)
        
        print(f"\nCHM-BASED SEGMENTATION COMPLETED!")
        print(f"Total trees found: {len(df)}")
        print(f"Average height: {df['max_height'].mean():.1f}m")
        print(f"Height range: {df['max_height'].min():.1f}m to {df['max_height'].max():.1f}m")
        print(f"Crown area range: {df['area_m2'].min():.1f}m² to {df['area_m2'].max():.1f}m²")
        print(f"Output: {output_csv}")
        
        return df
    else:
        print("No trees detected!")
        return None

if __name__ == "__main__":
    df_trees = run_chm_only_segmentation()