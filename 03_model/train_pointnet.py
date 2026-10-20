import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import open3d as o3d
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
TRAIN_CSV = r"E:\PointCloud_Tree_Modelling\dataset\tree_segmentation\tree_train.csv"
TEST_CSV  = r"E:\PointCloud_Tree_Modelling\dataset\tree_segmentation\tree_test.csv"
CHUNK_PATH = r"E:\PointCloud_Tree_Modelling\dataset\normalized\normalized_canopy_new"
NUM_POINTS = 4096
BATCH_SIZE = 4
EPOCHS = 50
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class TreeDataset(Dataset):
    def __init__(self, csv_path, chunk_dir, num_points=2048):
        self.df = pd.read_csv(csv_path)
        self.chunk_dir = chunk_dir
        self.num_points = num_points

        # Define metric columns (targets)
        self.target_cols = [
            'max_height', 'mean_height', 'area_m2', 
            'volume_approx', 'x_position', 'y_position', 'crown_radius'
        ]
        # Normalize metrics using MinMaxScaler
        self.scaler = MinMaxScaler()
        self.df[self.target_cols] = self.scaler.fit_transform(self.df[self.target_cols])
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        chunk_name = str(row['chunk_name']).strip()
        # Ensure correct filename format
        if not chunk_name.endswith(".ply"):
            chunk_file = f"normalized_canopy_spatial_{chunk_name}.ply"
        else:
            chunk_file = chunk_name

        full_path = os.path.join(self.chunk_dir, chunk_file)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Missing file: {full_path}")
        # Load point cloud
        pcd = o3d.io.read_point_cloud(full_path)
        pts = np.asarray(pcd.points)
        if pts.shape[0] == 0:
            pts = np.zeros((self.num_points, 3))
        else:
            choice = np.random.choice(pts.shape[0], self.num_points, replace=True)
            pts = pts[choice, :] 
        # Normalize coordinates
        pts = pts - np.mean(pts, axis=0)
        max_norm = np.max(np.linalg.norm(pts, axis=1))
        pts = pts / max_norm if max_norm > 0 else pts

        # Target metrics
        metrics = row[self.target_cols].values.astype(np.float32)
        return torch.tensor(pts, dtype=torch.float32), torch.tensor(metrics, dtype=torch.float32)

# POINTNET REGRESSOR MODEL
class TNet(nn.Module):
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        B = x.size(0)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        iden = torch.eye(self.k, device=x.device).view(1, self.k * self.k).repeat(B, 1)
        x = x + iden
        return x.view(-1, self.k, self.k)

class PointNetRegressor(nn.Module):
    def __init__(self, output_dim=7):
        super(PointNetRegressor, self).__init__()
        self.input_transform = TNet(k=3)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        B, N, _ = x.size()
        x = x.transpose(2, 1)
        trans = self.input_transform(x)
        x = torch.bmm(trans, x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return x

# TRAINING FUNCTION
def train_loop(train_csv, test_csv):
    train_dataset = TreeDataset(train_csv, CHUNK_PATH, NUM_POINTS)
    test_dataset = TreeDataset(test_csv, CHUNK_PATH, NUM_POINTS)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    model = PointNetRegressor(output_dim=7).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print("\n Training started...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for pts, metrics in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]"):
            pts, metrics = pts.to(DEVICE), metrics.to(DEVICE)
            optimizer.zero_grad()
            preds = model(pts)
            loss = criterion(preds, metrics)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {avg_loss:.4f}")

    # --- Evaluate ---
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for pts, metrics in test_loader:
            pts, metrics = pts.to(DEVICE), metrics.to(DEVICE)
            preds = model(pts)
            total_loss += criterion(preds, metrics).item()

    avg_test_loss = total_loss / len(test_loader)
    print(f"\n Final Test Loss: {avg_test_loss:.4f}")

    torch.save(model.state_dict(), "pointnet_tree_metrics.pth")
    print(" Model saved as pointnet_tree_metrics.pth")

if __name__ == "__main__":
    train_loop(TRAIN_CSV, TEST_CSV)
