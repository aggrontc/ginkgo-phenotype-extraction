import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ------------------------- 工具函数 -------------------------
def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid_xyz = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid_xyz) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def index_points(points, idx):
    B, N, C = points.shape
    if idx.dim() == 2:
        S = idx.shape[1]
        device = points.device
        batch_indices = torch.arange(B, dtype=torch.long, device=device).view(B, 1).repeat(1, S)
        return points[batch_indices, idx]
    elif idx.dim() == 3:
        B, S, K = idx.shape
        device = points.device
        batch_indices = torch.arange(B, dtype=torch.long, device=device).view(B, 1, 1).repeat(1, S, K)
        return points[batch_indices, idx]
    else:
        raise ValueError("Unsupported idx shape: expected 2 or 3 dims, got {}".format(idx.dim()))

def knn_point(k, xyz, new_xyz):
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    xyz_expand = xyz.unsqueeze(1).repeat(1, S, 1, 1)       # [B, S, N, 3]
    new_xyz_expand = new_xyz.unsqueeze(2).repeat(1, 1, N, 1)  # [B, S, N, 3]
    dist = torch.sum((xyz_expand - new_xyz_expand) ** 2, dim=3)  # [B, S, N]
    dist, idx = dist.topk(k, dim=2, largest=False, sorted=True)
    return dist, idx

def local_group(xyz, points, npoint, k=16):
    centroids = farthest_point_sample(xyz, npoint)        # [B, npoint]
    new_xyz = index_points(xyz, centroids)                # [B, npoint, 3]
    _, idx = knn_point(k, xyz, new_xyz)                   # idx: [B, npoint, k]
    grouped_xyz = index_points(xyz, idx)                  # [B, npoint, k, 3]
    grouped_points = index_points(points, idx)            # [B, npoint, k, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)
    grouped_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, k, C+3]
    return new_xyz, grouped_points

def interpolate_features(xyz1, xyz2, points1, points2, k=3):

    B, N1, _ = xyz1.shape
    _, N2, _ = xyz2.shape
    if N1 == 1:
        expanded_points = points1.repeat(1, N2, 1)
        return torch.cat([expanded_points, points2], dim=-1)
    dist, idx = knn_point(k, xyz1, xyz2)  # [B, N2, k]
    dist = torch.max(dist, torch.tensor([1e-10]).to(dist.device))
    weight = 1.0 / (dist + 1e-8)
    weight = weight / torch.sum(weight, dim=2, keepdim=True)
    interpolated_feats = torch.sum(index_points(points1, idx) * weight.unsqueeze(-1), dim=2)
    new_points = torch.cat([interpolated_feats, points2], dim=-1)
    return new_points

class InvResMLP(nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch, expand_ratio=2.0):
        super().__init__()
        ec = int(in_ch * expand_ratio)
        self.fc1 = nn.Linear(in_ch, ec)
        self.bn1 = nn.BatchNorm1d(ec)
        self.fc2 = nn.Linear(ec, ec)
        self.bn2 = nn.BatchNorm1d(ec)
        self.fc3 = nn.Linear(ec, out_ch)
        self.bn3 = nn.BatchNorm1d(out_ch)
        self.shortcut = nn.Sequential(
            nn.Linear(in_ch, out_ch),
            nn.BatchNorm1d(out_ch)
        ) if in_ch != out_ch else nn.Identity()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):  # x: [B,n,k,C]
        B,n,k,C = x.shape
        x = x.view(B*n*k, C)
        sc = self.shortcut(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.bn3(self.fc3(x))
        x = x + sc
        x = F.relu(x)
        x = self.dropout(x)
        return x.view(B, n, k, -1)

class PointSetAbstraction(nn.Module):
    def __init__(self, npoint, k, in_ch, out_ch):
        super().__init__()
        self.np, self.k = npoint, k
        self.inv = InvResMLP(in_ch+3, out_ch//2, out_ch)

    def forward(self, xyz, pts):
        new_xyz, grouped = local_group(xyz, pts, self.np, self.k)
        feats = self.inv(grouped)
        new_pts = feats.max(dim=2)[0]
        return new_xyz, new_pts

class PointFeaturePropagation(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_ch, out_ch), nn.BatchNorm1d(out_ch), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(out_ch, out_ch), nn.BatchNorm1d(out_ch), nn.ReLU()
        )

    def forward(self, xyz1, xyz2, pts1, pts2):
        new_pts = interpolate_features(xyz1, xyz2, pts1, pts2)
        B,N,C = new_pts.shape
        x = new_pts.view(B*N, C)
        x = self.mlp(x)
        return x.view(B, N, -1)

class PointNeXt(nn.Module):
    def __init__(self, num_classes=2, use_normal=True):
        super().__init__()
        in_ch = 6 if use_normal else 3
        self.sa1 = PointSetAbstraction(2048, 16, in_ch, 128)
        self.sa2 = PointSetAbstraction(512,  16, 128,   256)
        self.sa3 = PointSetAbstraction(128,  16, 256,   512)
        self.fp3 = PointFeaturePropagation(512+256, 256)
        self.fp2 = PointFeaturePropagation(256+128, 128)
        self.fp1 = PointFeaturePropagation(128+in_ch, 64)
        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        B,N,C = x.shape
        xyz = x[:,:,:3]
        pts = x
        xyz1, p1 = self.sa1(xyz, pts)
        xyz2, p2 = self.sa2(xyz1, p1)
        xyz3, p3 = self.sa3(xyz2, p2)
        l2 = self.fp3(xyz3, xyz2, p3, p2)
        l1 = self.fp2(xyz2, xyz1, l2, p1)
        l0 = self.fp1(xyz1, xyz, l1, x)
        out = self.classifier(l0.view(B*N, -1))
        return out.view(B, N, -1)

if __name__ == '__main__':
    # 测试模型输出
    model = PointNeXt(num_classes=2, use_normal=True)
    dummy_input = torch.randn(2, 2048, 6)  # batch=2, 每个点云2048个点，6维特征
    output = model(dummy_input)
    print("Output shape:", output.shape)  # 预期输出: [2, 2048, 2]
