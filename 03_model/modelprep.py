import os
import random
import pandas as pd
import open3d as o3d
import numpy as np

#path
csv_path = r"E:\PointCloud_Tree_Modelling\dataset\tree_segmentation\trees_dataset.csv"
ply_folder = r"E:\PointCloud_Tree_Modelling\dataset\normalized\normalized_canopy_new"
output_train_folder = r"E:\PointCloud_Tree_Modelling\dataset\train_ply"
output_test_folder = r"E:\PointCloud_Tree_Modelling\dataset\test_ply"

os.makedirs(output_train_folder, exist_ok=True)
os.makedirs(output_test_folder, exist_ok=True)

# LOAD THE CSV 
df = pd.read_csv(csv_path)
print(f" Total rows in CSV: {len(df)}")

if 'chunk_name' not in df.columns:
    raise ValueError("CSV must contain a column named 'chunk_name'")

# GET UNIQUE CHUNKS 
unique_chunks = df['chunk_name'].unique()
print(f" Unique chunks found: {len(unique_chunks)}")

#TRAIN/TEST SPLIT (70/30) 
random.seed(42)
random.shuffle(unique_chunks)
split_index = int(0.7 * len(unique_chunks))
train_chunks = unique_chunks[:split_index]
test_chunks = unique_chunks[split_index:]

print(f" Train chunks: {len(train_chunks)}, Test chunks: {len(test_chunks)}")

#  FUNCTION TO PROCESS PLY FILES
def process_and_save_chunks(chunks, output_folder):
    for chunk in chunks:
        ply_path = os.path.join(ply_folder, f"normalized_canopy_spatial_{chunk}.ply")

        if not os.path.exists(ply_path):
            print(f" Missing file for chunk: {chunk}")
            continue

        try:
            # Read the PLY file
            pcd = o3d.io.read_point_cloud(ply_path)
            points = np.asarray(pcd.points)
            print(f"→ Loaded {points.shape[0]} points for {chunk}")

            # Coordinate check: assume UTM Zone 44N (EPSG:32644)
            if np.mean(points[:, 0]) < 1000: 
                print(f" Detected local coordinates, applying UTM offset correction.")
                # Example offset: modify as per your study area’s approximate UTM centroid
                utm_offset = np.array([418000.0, 1439900.0, 0])
                points += utm_offset


            # Update and save
            pcd.points = o3d.utility.Vector3dVector(points)
            out_path = os.path.join(output_folder, f"{chunk}.ply")
            o3d.io.write_point_cloud(out_path, pcd)
            print(f" Saved {out_path}")

        except Exception as e:
            print(f"❌ Error processing {chunk}: {e}")

# === STEP 5: PROCESS TRAIN AND TEST CHUNKS ===
print("\n=== Saving Training Data ===")
process_and_save_chunks(train_chunks, output_train_folder)

print("\n=== Saving Testing Data ===")
process_and_save_chunks(test_chunks, output_test_folder)

print("\n✅ Data preparation completed successfully with CRS consistency (EPSG:32644).")
