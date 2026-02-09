# ISVS/src/optimizer.py
import time
import math
import csv
from pathlib import Path
import torch

def _mat_cond(H: torch.Tensor):
    try:
        s = torch.linalg.svdvals(H)
        c = (s.max() / s.min()).item()
        return float(c)
    except Exception:
        return float("nan")

def gauss_newton_minimize(
    f, z0: torch.Tensor,
    max_iter=15,
    lm_damping: float = 0.0,
    ls_backtrack: bool = True,
    verbose: bool = True,
    log_every: int = 1,
    save_history_csv: str | None = None,
):
    """
    f: z -> (r,J,aux)  を返す関数
      - r: (K,1)
      - J: (K,8)
      - aux: 任意（ここでは inbounds 等のデバッグ情報を想定）
    """

    def _energy(r):  # 0.5 * ||r||^2
        return 0.5 * float((r * r).sum().item())

    z = z0.detach().clone().float()
    hist = {"iters": 0, "E0": 0.0, "Efin": 0.0}
    t0 = time.time()

    r, J, aux = f(z)
    E = _energy(r)
    hist["E0"] = E

    # CSV準備
    writer = None
    if save_history_csv:
        p = Path(save_history_csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        fcsv = p.open("w", newline="", encoding="utf-8")
        writer = csv.writer(fcsv)
        writer.writerow([
            "iter","E","dE","alpha","norm_dz","condH","rankH","infGrad",
            "res_min","res_max","K_valid","K_total","sec"
        ])
    else:
        fcsv = None

    if verbose:
        print("iter |     E         dE        alpha    ||dz||     cond(H)    rank(H)   ||g||inf   r[min,max]     Kvalid/K   time[s]")
        print("-"*115)
        print(f"{0:4d} | {E:10.6f} {' ':11} {' ':9} {' ':8} {' ':10} {' ':9} {' ':9} "
              f"[{r.min().item():.4f},{r.max().item():.4f}] "
              f"{aux.get('K_valid', r.shape[0])}/{aux.get('K_total', r.shape[0])} {0:8.3f}")

    for it in range(1, max_iter+1):
        t_iter0 = time.time()

        JT = J.T
        H = JT @ J                      # (8,8)
        g = JT @ r                      # (8,1)
        if lm_damping and lm_damping > 0:
            H = H + lm_damping * torch.eye(H.shape[0], dtype=H.dtype)

        # 解析量（ログ用）
        try:
            rankH = int(torch.linalg.matrix_rank(H).item())
        except Exception:
            rankH = -1
        condH = _mat_cond(H)
        infGrad = float(torch.linalg.norm(g, ord=float("inf")).item())

        # Δz 解く
        try:
            dz = torch.linalg.solve(H, -g).squeeze(1)  # (8,)
        except Exception:
            dz = torch.linalg.lstsq(H, -g).solution.squeeze(1)

        # 直線探索
        alpha = 1.0
        z_new = z + alpha * dz
        r_new, J_new, aux_new = f(z_new)
        E_new = _energy(r_new)

        if ls_backtrack and E_new > E:          # 単純バックトラック
            ok = False
            for _ in range(8):
                alpha *= 0.5
                z_new = z + alpha * dz
                r_new, J_new, aux_new = f(z_new)
                E_new = _energy(r_new)
                if E_new <= E:
                    ok = True
                    break
            if not ok:
                # 収束しないので終了（更新拒否）
                break

        # 受理
        dE = E - E_new
        z, r, J, aux, E = z_new, r_new, J_new, aux_new, E_new
        hist["iters"] = it
        t_iter = time.time() - t_iter0

        # ログ出力
        if verbose and (it % log_every == 0):
            Ktot = aux.get("K_total", r.shape[0])
            Kval = aux.get("K_valid", r.shape[0])
            print(f"{it:4d} | {E:10.6f} {dE:11.6f} {alpha:9.3f} {float(torch.linalg.norm(dz).item()):8.3f} "
                  f"{condH:10.3e} {rankH:9d} {infGrad:9.3e} "
                  f"[{r.min().item():.4f},{r.max().item():.4f}] "
                  f"{Kval}/{Ktot} {t_iter:8.3f}")

        # CSV記録
        if writer:
            Ktot = aux.get("K_total", r.shape[0])
            Kval = aux.get("K_valid", r.shape[0])
            writer.writerow([
                it, E, dE, alpha, float(torch.linalg.norm(dz).item()),
                condH, rankH, infGrad,
                r.min().item(), r.max().item(), Kval, Ktot, t_iter
            ])

        # 小さすぎる更新で停止（任意）
        if float(torch.linalg.norm(dz).item()) < 1e-8:
            break

    if fcsv:
        fcsv.close()

    hist["Efin"] = E
    hist["sec"] = time.time() - t0
    return z, hist
