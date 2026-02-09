# ISVS/src/objective.py
import torch
import numpy as np
from typing import Tuple
from src.homography import lie_to_H, warp_points_homography

def _bilinear_sample_feat_grid(gf, xy_obs: torch.Tensor) -> torch.Tensor:
    """
    観測側特徴マップ (gh,gw,D) を表示座標でバイリニア補間。
    入力 xy_obs: (K,2) in [0, Wc)×[0, Hc)
    """
    F = gf.patch_feat    # (N,D)
    gh, gw = gf.gh, gf.gw
    Wc, Hc = gf.shown_size
    D = F.shape[1]
    Fg = F.view(gh, gw, D)  # (gh,gw,D)

    # 画像座標 → グリッド座標（セル中心が整数+0.5）
    px_x = Wc / gw
    px_y = Hc / gh
    qx_f = xy_obs[:,0] / px_x - 0.5
    qy_f = xy_obs[:,1] / px_y - 0.5

    # 4近傍
    x0 = torch.clamp(qx_f.floor().long(), 0, gw-1)
    y0 = torch.clamp(qy_f.floor().long(), 0, gh-1)
    x1 = torch.clamp(x0 + 1, 0, gw-1)
    y1 = torch.clamp(y0 + 1, 0, gh-1)

    # 重み
    fx = (qx_f - x0.float()).clamp(0,1)
    fy = (qy_f - y0.float()).clamp(0,1)

    def gather(xi, yi):
        return Fg[yi, xi, :]  # (K,D)

    f00 = gather(x0, y0)
    f10 = gather(x1, y0)
    f01 = gather(x0, y1)
    f11 = gather(x1, y1)

    f0 = f00 * (1 - fx).unsqueeze(1) + f10 * fx.unsqueeze(1)
    f1 = f01 * (1 - fx).unsqueeze(1) + f11 * fx.unsqueeze(1)
    f  = f0  * (1 - fy).unsqueeze(1) + f1  * fy.unsqueeze(1)

    # L2正規化
    f = torch.nn.functional.normalize(f, dim=1)
    return f  # (K,D)

def residuals_cosine(Ft_sel: torch.Tensor, Fc_warp: torch.Tensor) -> torch.Tensor:
    # r_k = 1 - <Fc_warp[k], Ft_sel[k]>
    sim = (Fc_warp * Ft_sel).sum(dim=1)  # (K,)
    r = 1.0 - sim
    return r.unsqueeze(1)                # (K,1)

def numeric_jacobian(fun, z: torch.Tensor, eps: float = 1e-4):
    """
    中心差分ヤコビアン。fun(z) → r:(K,1)
    戻り値: J:(K,8)
    """
    z = z.detach().clone().float()
    r0 = fun(z)            # (K,1)
    K = r0.shape[0]
    J = torch.zeros((K, z.numel()), dtype=torch.float32)
    for i in range(z.numel()):
        dz = torch.zeros_like(z); dz[i] = eps
        rp = fun(z + dz)
        rm = fun(z - dz)
        J[:, i] = ((rp - rm) / (2*eps)).squeeze(1)
    return r0, J

def _inbounds_mask(xy: torch.Tensor, W: int, H: int):
    x = xy[:,0]; y = xy[:,1]
    return (x >= 0) & (x <= (W-1)) & (y >= 0) & (y <= (H-1))

def build_objective(Ft: torch.Tensor, gf_c, P_t: torch.Tensor, picks):
    idxs = torch.tensor([idx for (_,_,idx) in picks], dtype=torch.long)
    Ft_sel = torch.nn.functional.normalize(Ft[idxs, :].float(), dim=1)  # (K,D)
    Wc, Hc = gf_c.shown_size

    def f_core(z: torch.Tensor):
        G = lie_to_H(z.float())
        xy_o = warp_points_homography(G, P_t)                  # (K,2)

        m = _inbounds_mask(xy_o, W=Wc, H=Hc)
        K_total = int(xy_o.shape[0])
        K_valid = int(m.sum().item())
        if K_valid == 0:
            # 全て画外：ダミーを返す（大きな残差で誘導）
            r = torch.ones((K_total,1), dtype=torch.float32)
            aux = {"G": G, "xy_o": xy_o, "K_total": K_total, "K_valid": K_valid}
            return r, aux

        # 必要に応じて out-of-bounds を無視する（最初は全点使いたいなら m を無視してOK）
        xy_in  = xy_o[m]
        Ft_in  = Ft_sel[m]

        Fc_w = _bilinear_sample_feat_grid(gf_c, xy_in)         # (Kv,D)
        r_in = residuals_cosine(Ft_in, Fc_w)                   # (Kv,1)

        # 出力の次元は元のKに合わせる（画外は残差=1に）
        r = torch.ones((K_total,1), dtype=torch.float32)
        r[m] = r_in

        aux = {"G": G, "xy_o": xy_o, "K_total": K_total, "K_valid": K_valid}
        return r, aux

    def f(z: torch.Tensor):
        r, aux = f_core(z)
        # 中心差分ヤコビアン
        def fun_only(z_): 
            return f_core(z_)[0]
        r0, J = numeric_jacobian(fun_only, z, eps=1e-4)
        return r0, J, aux

    return f
