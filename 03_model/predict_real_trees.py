import torch
import torch.nn as nn
import numpy as np
import open3d as o3d
from sklearn.preprocessing import MinMaxScaler
import joblib  # For saving/loading the scaler (optional)

# MODEL DEFINITIONS

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

# PREDICTION PIPELINE
def predict_single_tree(ply_path, model_path, scaler_path=None, num_points=2048):
    # Load trained model
    model = PointNetRegressor(output_dim=7)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # Load PLY file
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points)
    if pts.shape[0] == 0:
        raise ValueError(f"Empty point cloud: {ply_path}")

    # Random sampling
    choice = np.random.choice(pts.shape[0], num_points, replace=True)
    pts = pts[choice, :]

    # Normalize
    pts = pts - np.mean(pts, axis=0)
    max_norm = np.max(np.linalg.norm(pts, axis=1))
    pts = pts / max_norm if max_norm > 0 else pts

    # Convert to tensor
    pts = torch.tensor(pts, dtype=torch.float32).unsqueeze(0)  # [1, N, 3]

    # Predict
    with torch.no_grad():
        preds = model(pts).cpu().numpy().flatten()

    feature_names = ['max_height', 'mean_height', 'area_m2', 'volume_approx',
                     'x_position', 'y_position', 'crown_radius']

    pred_dict = {k: v for k, v in zip(feature_names, preds)}
    # Reverse scale (optional)
    if scaler_path:
        scaler = joblib.load(scaler_path)
        unscaled = scaler.inverse_transform([preds])[0]
        pred_dict = {k: v for k, v in zip(feature_names, unscaled)}
    return pred_dict
# MAIN
if __name__ == "__main__":
    MODEL_PATH = r"E:\PointCloud_Tree_Modelling\pointnett\pointnet_tree_metrics_regressor.pth"
    TEST_PLY = r"E:\PointCloud_Tree_Modelling\dataset\normalized\normalized_canopy_new\normalized_canopy_spatial_chunk_25.ply"
    
    result = predict_single_tree(TEST_PLY, MODEL_PATH)
    print("\nPredicted Tree Metrics:")
    for k, v in result.items():
        print(f"{k:15s}: {v:.4f}")

