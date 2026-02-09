# ISVS/src/homography.py
import torch
import numpy as np

def _basis_matrices_sl3():
    # 8基底（平行移動・回転・スケール・せん断・射影）。一例。
    # 参考: Hartley-Zisserman系／SL(3)の典型基底
    A = []
    A.append(torch.tensor([[0,0,1],[0,0,0],[0,0,0]], dtype=torch.float32))   # tx
    A.append(torch.tensor([[0,0,0],[0,0,1],[0,0,0]], dtype=torch.float32))   # ty
    A.append(torch.tensor([[0,-1,0],[1,0,0],[0,0,0]], dtype=torch.float32))  # rot (小回転)
    A.append(torch.tensor([[1,0,0],[0,-1,0],[0,0,0]], dtype=torch.float32))  # iso scale (trace=0)
    A.append(torch.tensor([[0,1,0],[0,0,0],[0,0,0]], dtype=torch.float32))   # shear xy
    A.append(torch.tensor([[0,0,0],[1,0,0],[0,0,0]], dtype=torch.float32))   # shear yx
    A.append(torch.tensor([[0,0,0],[0,0,0],[1,0,0]], dtype=torch.float32))   # proj x
    A.append(torch.tensor([[0,0,0],[0,0,0],[0,1,0]], dtype=torch.float32))   # proj y
    return A

_A = _basis_matrices_sl3()

def lie_to_H(z: torch.Tensor) -> torch.Tensor:
    """z∈R^8 → H=exp(sum z_i A_i) ∈ SL(3)"""
    assert z.numel() == 8
    M = torch.zeros((3,3), dtype=torch.float32)
    for i in range(8):
        M = M + z[i] * _A[i]
    H = torch.matrix_exp(M)
    return H

def warp_points_homography(G: torch.Tensor, pts_xy: torch.Tensor) -> torch.Tensor:
    """(K,2) → (K,2) 同次→正規化"""
    K = pts_xy.shape[0]
    ones = torch.ones((K,1), dtype=torch.float32)
    P = torch.cat([pts_xy, ones], dim=1).T  # (3,K)
    Q = G @ P                               # (3,K)
    qx = Q[0,:] / (Q[2,:] + 1e-8)
    qy = Q[1,:] / (Q[2,:] + 1e-8)
    return torch.stack([qx, qy], dim=1)

def grid_centers_t(Wt: int, Ht: int, gw_t: int, gh_t: int, picks):
    """target表示座標の格子中心 (K,2)"""
    px_x = Wt / gw_t
    px_y = Ht / gh_t
    pts = []
    for (qy, qx, _) in picks:
        x = (qx + 0.5) * px_x
        y = (qy + 0.5) * px_y
        pts.append((x, y))
    return torch.tensor(pts, dtype=torch.float32)

def polygon_from_image_size(W: int, H: int):
    return np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype=np.float32)
