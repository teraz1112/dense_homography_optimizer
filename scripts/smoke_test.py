from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "outputs" / "smoke"
    if out_dir.exists():
        shutil.rmtree(out_dir)

    cmd = [
        sys.executable,
        "dense_ibvs.py",
        "--encoder",
        "dummy",
        "--target",
        "data/samples/target.png",
        "--observed",
        "data/samples/observed.png",
        "--mask",
        "data/samples/mask.png",
        "--out-dir",
        str(out_dir),
        "--run-name",
        "smoke_dense",
        "--max-iter",
        "3",
    ]
    subprocess.run(cmd, cwd=root, check=True)

    expected = out_dir / "smoke_dense_H_overlay.png"
    if not expected.exists():
        raise SystemExit(f"Smoke test failed: missing {expected}")
    print(f"[OK] Smoke test passed: {expected}")


if __name__ == "__main__":
    main()
