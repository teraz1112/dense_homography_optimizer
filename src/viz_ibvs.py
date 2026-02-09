# ISVS/src/viz_ibvs.py
import numpy as np
from PIL import Image, ImageDraw
import torch
from src.homography import polygon_from_image_size, warp_points_homography

def draw_homography_on_observed(observed_img: Image.Image, G: torch.Tensor,
                                color=(0,255,0), thick=3) -> Image.Image:
    W, H = observed_img.size
    poly_t = polygon_from_image_size(W, H)              # target矩形（表示系で同サイズとみなして描く）
    pts = torch.tensor(poly_t, dtype=torch.float32)
    pts_w = warp_points_homography(G, pts)              # (4,2)
    pts_w_np = pts_w.numpy().clip([0,0], [W-1, H-1])

    im = observed_img.copy()
    draw = ImageDraw.Draw(im)
    seq = pts_w_np.tolist() + [pts_w_np[0].tolist()]
    for i in range(4):
        draw.line([tuple(seq[i]), tuple(seq[i+1])], fill=color, width=thick)
    return im

def polygon_from_image_size(W: int, H: int):
    return np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype=np.float32)

# 末尾に追記
import matplotlib.pyplot as plt

def show_side_by_side(target_img: Image.Image, observed_overlay: Image.Image,
                      title_left="Target (generated)", title_right="Observed + H",
                      save_path: str | None = None, block: bool = True):
    """生成画像と推定Hを重畳した観測画像を横並びで表示。必要なら保存。"""
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.imshow(target_img)
    ax1.set_title(title_left)
    ax1.axis("off")

    ax2.imshow(observed_overlay)
    ax2.set_title(title_right)
    ax2.axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show(block=block)
