# Dense Homography Optimizer

## 1) これは何か
- `dense_ibvs.py` で Dense Homography 最適化を実行
- マスク内パッチの特徴残差を最小化し、観測画像上に推定ホモグラフィを重畳
- 反復履歴CSVと可視化画像を保存

## 2) 前提（OS, GPU/CPU, 主要依存）
- 想定OS: Windows/Linux
- Python: 3.10+
- 主要依存: `torch`, `transformers`, `numpy`, `opencv-python`, `Pillow`, `matplotlib`
- CPUでも実行可能（速度はGPU推奨）

## 3) セットアップ（最短手順）
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

## 4) 最小実行例（コピペ可能）
```bash
python dense_ibvs.py --target data/samples/target.png --observed data/samples/DG/{DGtype}.png --mask data/samples/mask.png --out-dir outputs/demo --run-name dense_ibvs
```

設定ファイル利用:
```bash
python dense_ibvs.py --config configs/default.yaml
```

## 5) データの置き場所（サンプル、取得方法）
- 最小サンプル: `data/samples/target.png`, `data/samples/DG/{DGtype}.png`, `data/samples/mask.png`
- 出力: `--out-dir` 配下
  - `<run_name>_H_overlay.png`
  - `<run_name>_side_by_side.png`
  - `<run_name>_history.csv`

## 6) よくあるエラーと対処
- `Target/Observed/Mask image not found`
  - 相対パスの基準はこのディレクトリ
- `Mask produced zero picks`
  - マスク領域が狭すぎる。`--mask-min-cover` を下げる
- モデルDLが重い
  - 検証目的なら `--encoder dummy` で軽量実行可能

## 7) プロジェクト構造（各ディレクトリの役割）
```text
dense_homography_optimizer/
├─ dense_ibvs.py            # CLIエントリ
├─ configs/default.yaml     # 実行設定例
├─ src/                     # 最適化と可視化ロジック
├─ data/samples/            # 最小サンプル入力
└─ scripts/smoke_test.py    # スモークテスト
```

## 8) ライセンス/引用（third_party含む）
- このディレクトリには外部リポジトリのソース同梱はありません
- 使用ライブラリ・モデルのライセンスは各配布元に従ってください
