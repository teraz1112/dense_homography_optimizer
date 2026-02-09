from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

from src.cropper import crop_to_match
from src.homography import grid_centers_t, lie_to_H
from src.io_tools import ensure_outdir, load_image, save_image
from src.mask_to_picks import mask_to_picks
from src.objective import build_objective
from src.optimizer import gauss_newton_minimize
from src.viz_ibvs import draw_homography_on_observed, show_side_by_side


def _load_yaml(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class DummyGridFeatures:
    patch_feat: torch.Tensor
    gh: int
    gw: int
    shown_size: tuple[int, int]


class DummyExtractor:
    def extract(self, img: Image.Image, vit_patch: int, long_side: int, pad_to_multiple: bool):
        if long_side > 0:
            w, h = img.size
            scale = long_side / max(w, h)
            new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
            img_r = img.resize(new_size, Image.BICUBIC)
        else:
            img_r = img.copy()

        w, h = img_r.size
        if pad_to_multiple:
            w_pad = ((w + vit_patch - 1) // vit_patch) * vit_patch
            h_pad = ((h + vit_patch - 1) // vit_patch) * vit_patch
            if (w_pad, h_pad) != (w, h):
                canvas = Image.new("RGB", (w_pad, h_pad))
                canvas.paste(img_r, (0, 0))
                img_x = canvas
            else:
                img_x = img_r
        else:
            img_x = img_r

        arr = np.asarray(img_x).astype(np.float32) / 255.0
        gh = arr.shape[0] // vit_patch
        gw = arr.shape[1] // vit_patch
        feats = []
        for qy in range(gh):
            for qx in range(gw):
                y0 = qy * vit_patch
                y1 = y0 + vit_patch
                x0 = qx * vit_patch
                x1 = x0 + vit_patch
                patch = arr[y0:y1, x0:x1, :]
                mean_rgb = patch.mean(axis=(0, 1))
                std_rgb = patch.std(axis=(0, 1))
                pos = np.array([(qx + 0.5) / gw, (qy + 0.5) / gh], dtype=np.float32)
                f = np.concatenate([mean_rgb, std_rgb, pos], axis=0)  # D=8
                feats.append(f)
        feat = torch.tensor(np.asarray(feats), dtype=torch.float32)
        feat = torch.nn.functional.normalize(feat, dim=-1)
        return DummyGridFeatures(patch_feat=feat, gh=gh, gw=gw, shown_size=img_r.size)


def _build_extractor(args):
    if args.encoder == "dummy":
        return DummyExtractor()
    from src.encoders import get_feature_extractor

    extractor = get_feature_extractor(
        args.encoder,
        args.model_name,
        feat_mode=args.feat_mode,
        feat_layer=args.feat_layer,
        feat_last_k=args.feat_last_k,
    )
    if args.encoder == "ours":
        extractor.shallow_layer = int(args.ours_shallow_layer)
        extractor.clip_feat_mode = args.ours_clip_feat_mode
        extractor.clip_feat_layer = int(args.ours_clip_feat_layer)
        extractor.clip_feat_last_k = int(args.ours_clip_feat_last_k)
        extractor.use_dino_deep = bool(args.ours_use_dino_deep)
        extractor.use_dino_shallow = bool(args.ours_use_dino_shallow)
        extractor.use_clip = bool(args.ours_use_clip)
    return extractor


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dense homography optimization.")
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config.")
    parser.add_argument("--target", type=Path, default=Path("data/samples/target.png"))
    parser.add_argument("--observed", type=Path, default=Path("data/samples/observed.png"))
    parser.add_argument("--mask", type=Path, default=Path("data/samples/mask.png"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--run-name", type=str, default="dense_ibvs")

    parser.add_argument("--encoder", type=str, default="dino", choices=["dino", "clip", "ours", "dummy"])
    parser.add_argument("--model-name", type=str, default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--vit-patch", type=int, default=16)
    parser.add_argument("--long-side", type=int, default=1600)
    parser.add_argument("--pad-to-multiple", action="store_true")
    parser.add_argument("--no-pad-to-multiple", action="store_false", dest="pad_to_multiple")
    parser.set_defaults(pad_to_multiple=True)
    parser.add_argument("--feat-mode", type=str, default="last")
    parser.add_argument("--feat-layer", type=int, default=-1)
    parser.add_argument("--feat-last-k", type=int, default=4)

    parser.add_argument("--ours-shallow-layer", type=int, default=6)
    parser.add_argument("--ours-clip-feat-mode", type=str, default="layer")
    parser.add_argument("--ours-clip-feat-layer", type=int, default=6)
    parser.add_argument("--ours-clip-feat-last-k", type=int, default=4)
    parser.add_argument("--ours-use-dino-deep", action="store_true")
    parser.add_argument("--ours-use-dino-shallow", action="store_true")
    parser.add_argument("--ours-use-clip", action="store_true")
    parser.set_defaults(ours_use_dino_deep=True, ours_use_dino_shallow=False, ours_use_clip=True)

    parser.add_argument("--crop-mode", type=str, default="scale_then_center_crop")
    parser.add_argument("--mask-min-cover", type=float, default=0.0)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--lm-lambda", type=float, default=1e-4)
    parser.add_argument("--ls-backtrack", action="store_true")
    parser.add_argument("--no-ls-backtrack", action="store_false", dest="ls_backtrack")
    parser.set_defaults(ls_backtrack=True)
    return parser


def _apply_yaml_defaults(args):
    cfg = _load_yaml(args.config).get("run", {})
    for key, val in cfg.items():
        if hasattr(args, key):
            setattr(args, key, val)
    return args


def main():
    parser = _build_parser()
    args = _apply_yaml_defaults(parser.parse_args())
    args.target = Path(args.target)
    args.observed = Path(args.observed)
    args.mask = Path(args.mask)
    args.out_dir = Path(args.out_dir)

    ensure_outdir(args.out_dir)
    if not args.target.exists():
        raise FileNotFoundError(f"Target image not found: {args.target}")
    if not args.observed.exists():
        raise FileNotFoundError(f"Observed image not found: {args.observed}")
    if not args.mask.exists():
        raise FileNotFoundError(f"Mask image not found: {args.mask}")

    img_t = load_image(args.target)
    img_o = load_image(args.observed)
    img_m = load_image(args.mask)

    cropped, _ = crop_to_match(target_img=img_t, observed_img=img_o, mode=args.crop_mode)
    save_image(cropped, args.out_dir / f"{args.run_name}_cropped.png", overwrite=True)

    extractor = _build_extractor(args)
    gf_t = extractor.extract(img_t, vit_patch=args.vit_patch, long_side=args.long_side, pad_to_multiple=args.pad_to_multiple)
    gf_o = extractor.extract(cropped, vit_patch=args.vit_patch, long_side=args.long_side, pad_to_multiple=args.pad_to_multiple)

    picks = mask_to_picks(mask_img=img_m, gf_t=gf_t, min_covered_ratio=float(args.mask_min_cover))
    if len(picks) == 0:
        raise RuntimeError("Mask produced zero picks. Adjust mask or threshold.")

    P_t = grid_centers_t(Wt=gf_t.shown_size[0], Ht=gf_t.shown_size[1], gw_t=gf_t.gw, gh_t=gf_t.gh, picks=picks)
    f_obj = build_objective(Ft=gf_t.patch_feat, gf_c=gf_o, P_t=P_t, picks=picks)
    z0 = torch.zeros(8, dtype=torch.float32)

    history_csv = args.out_dir / f"{args.run_name}_history.csv"
    z, hist = gauss_newton_minimize(
        f=f_obj,
        z0=z0,
        max_iter=int(args.max_iter),
        lm_damping=float(args.lm_lambda),
        ls_backtrack=bool(args.ls_backtrack),
        verbose=True,
        log_every=1,
        save_history_csv=str(history_csv),
    )
    print(f"[RESULT] iters={hist['iters']}, E0={hist['E0']:.6f}, Efin={hist['Efin']:.6f}, time={hist.get('sec', 0):.3f}s")

    G = lie_to_H(z)
    obs_disp = cropped.resize(gf_o.shown_size, Image.BICUBIC)
    overlay = draw_homography_on_observed(observed_img=obs_disp, G=G, color=(0, 255, 0), thick=3)
    out_path = args.out_dir / f"{args.run_name}_H_overlay.png"
    overlay.save(out_path)

    tgt_disp = img_t.resize(gf_t.shown_size, Image.BICUBIC)
    side_path = args.out_dir / f"{args.run_name}_side_by_side.png"
    show_side_by_side(
        target_img=tgt_disp,
        observed_overlay=overlay,
        title_left="Target (generated)",
        title_right="Observed + estimated H",
        save_path=str(side_path),
        block=False,
    )
    print(f"[OK] Saved: {out_path}")
    print(f"[OK] Saved: {side_path}")
    print(f"[OK] Saved: {history_csv}")


if __name__ == "__main__":
    main()
