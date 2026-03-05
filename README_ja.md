# リポジトリ詳細解説：大規模建物高さ推定（T-SwinUNet）

## 概要

このリポジトリは、**Sentinel-1 SAR（合成開口レーダー）** と **Sentinel-2 MSI（多波長画像）** の時系列データを組み合わせ、深層学習によって大規模な建物高さを推定するプロジェクトです。

提案モデル **T-SwinUNet**（Temporal Swin Transformer U-Net）は、12ヶ月分の衛星画像時系列を入力として受け取り、10m空間解像度で建物の高さをピクセル単位で予測します。

### 研究成果の要点

| 指標 | 値 |
|------|-----|
| RMSE（10m解像度・学習データ内） | **1.89 m** |
| RMSE（分布外テスト・他の欧州10都市） | **3.2 m** |
| RMSE（100m解像度） | **0.29 m** |
| R²（100m解像度） | **0.75** |

> **論文掲載誌：** Remote Sensing of Environment（Elsevier、2025年）  
> **DOI：** https://doi.org/10.1016/j.rse.2024.114556

---

## リポジトリ構成

```
sar2hight-original/
├── README.md                      # 英語版README（オリジナル）
├── README_ja.md                   # 本ファイル（日本語解説）
└── TSwinUnet/                     # メインプロジェクトディレクトリ
    ├── train.py                   # 学習スクリプト（エントリポイント）
    ├── predict.py                 # 推論スクリプト（エントリポイント）
    ├── environment.yml            # Conda環境定義ファイル
    ├── configs/
    │   └── tswin_unet/
    │       └── exp3.yaml          # メイン実験設定ファイル
    ├── assets/
    │   └── figures/               # 論文掲載図・可視化結果
    │       ├── dataset_location.png   # データセット地理的位置
    │       ├── Quant.jpg              # 定量評価結果
    │       ├── COR.jpg                # 相関プロット
    │       └── GEE_vis.jpg            # Google Earth Engine可視化
    └── libs/                      # コアライブラリ
        ├── models/                # モデル定義
        │   ├── tswin_unet/
        │   │   ├── tswin_unet.py  # T-SwinUNetアーキテクチャ本体
        │   │   └── swin.py        # Swin Transformerブロック実装
        │   └── __init__.py
        ├── train/                 # 学習関連
        │   ├── trainer.py         # 学習ループ・モデル管理
        │   ├── dataset.py         # データ読み込み・前処理
        │   └── __init__.py
        ├── predict/               # 推論関連
        │   ├── predictor.py       # 推論クラス
        │   └── __init__.py
        ├── utils/                 # 汎用ユーティリティ
        │   ├── base.py            # ベーストレーナークラス
        │   ├── logger.py          # ロギングユーティリティ
        │   ├── losses.py          # 損失関数（15種類以上）
        │   ├── scheduler.py       # 学習率スケジューラ
        │   ├── utils.py           # 汎用関数
        │   └── __init__.py
        └── process/               # データ処理
            ├── utils.py           # ラスターデータ処理ユーティリティ
            └── __init__.py
```

---

## モデルアーキテクチャ：T-SwinUNet

### 全体の処理フロー

```
入力: (B, 12, 9, 128, 128)
     └── B: バッチサイズ
     └── 12: 月数（1月〜12月）
     └── 9: チャンネル数（S1: 4バンド + S2: 5バンド）
     └── 128×128: パッチサイズ（ピクセル）

  ↓
[EfficientNet-B5 エンコーダ]
  ├── 特徴マップをマルチスケールで抽出
  └── Swin Transformerブロックで空間的特徴を強化

  ↓
[LTAE2d — 軽量時間的注意エンコーダ]
  └── 12ヶ月の時系列データに対して時間的注意機構を適用
  └── どの月が建物高さ推定に重要かを自動学習

  ↓
[U-Netデコーダ]
  └── スキップ接続でエンコーダの特徴マップを再利用
  └── アップサンプリングで元の解像度に復元

  ↓
出力（3系統）:
  ├── pred   : 建物高さ予測値（連続値、単位: m）
  ├── seg    : 建物セグメンテーションマスク（0/1）
  └── rseg   : 逆セグメンテーションマスク（境界精緻化）
```

### 主要コンポーネント

#### `libs/models/tswin_unet/tswin_unet.py`
T-SwinUNetのメインアーキテクチャが定義されています。主なクラス：
- **PositionalEncoder**：時系列データへの位置エンコーディング付与
- **LTAE2d**：2D画像用の軽量時間的注意機構（12ヶ月分を集約）
- **TSwinUnet**：モデル全体を統合するメインクラス

#### `libs/models/tswin_unet/swin.py`
Swin Transformerブロックの実装です。特徴：
- **ウィンドウアテンション v1/v2**：局所ウィンドウ内でのself-attention計算
- **パッチマージング**：階層的な特徴マップのダウンサンプリング
- **3Dパッチ埋め込み**：(時間, 高さ, 幅) の3次元入力に対応

---

## 入力データの詳細

### Sentinel-1 SAR データ（4バンド）
| バンド | 説明 |
|--------|------|
| VV（垂直送信/垂直受信） | 建物の後方散乱強度 |
| VH（垂直送信/水平受信） | 植生・建物の後方散乱強度 |
| その他2バンド | 追加SAR特徴量 |

### Sentinel-2 MSI データ（11バンド→5バンド使用）
可視光・近赤外・短波赤外域の多波長データを使用。

### 時系列構成
- **12ヶ月分**（1月〜12月）の観測データを1サンプルとして使用
- 各月ごとにSentinel-1・Sentinel-2をスタック
- 欠損データは0パディングで補完（マスクフラグで管理）

### 参照データ（Ground Truth）
- `LABEL_1m/`：1m解像度の建物高さラベル（GeoTIFF形式）
- 建物が存在しないピクセルは高さ0として扱う
- 1m未満の高さは非建物として0に設定

---

## データセット

使用データセット：**M4Heights**  
🔗 [Hugging Face: Rituxx96x/M4Heights](https://huggingface.co/datasets/Rituxx96x/M4Heights)

### 学習・評価地域
| 役割 | 地域 |
|------|------|
| 学習データ | オランダ、スイス、エストニア、ドイツ |
| 分布外テスト（OOD） | ヨーロッパ他10都市 |

### データ構造（想定）
```
データルートディレクトリ/
├── LABEL_1m/         # 建物高さラベル（GeoTIFF）
├── S1/               # Sentinel-1 SAR データ（月別TIF）
│   ├── patch_001_00.tif  # パッチ001 1月
│   ├── patch_001_01.tif  # パッチ001 2月
│   └── ...
└── S2/               # Sentinel-2 MSI データ（月別TIF）
    ├── patch_001_00.tif
    └── ...
```

CSVファイル（学習リスト）形式：
```csv
fname
patch_001.tif
patch_002.tif
...
```

---

## 環境構築

### 前提条件
- CUDA 11.8対応のGPU
- Anaconda または Miniconda

### インストール手順

```bash
# リポジトリのクローン
git clone <リポジトリURL>
cd sar2hight-original/TSwinUnet

# Conda環境の作成（300以上のパッケージを含む）
conda env create -f environment.yml

# 環境の有効化
conda activate <環境名>
```

### 主要な依存ライブラリ

| カテゴリ | ライブラリ | バージョン | 用途 |
|---------|-----------|---------|------|
| 深層学習 | PyTorch | 2.0.0 | ニューラルネットワーク学習 |
| 深層学習 | torchvision | 0.15.1 | 画像モデルユーティリティ |
| 医療画像 | MONAI | 1.1.0 | UNetデコーダブロック |
| ビジョンTransformer | timm | 0.6.13 | EfficientNetエンコーダ |
| 地理空間 | rasterio | 1.3.6 | GeoTIFF読み書き |
| 地理空間 | geopandas | 0.13.0 | ベクターデータ処理 |
| 画像処理 | OpenCV | 4.7.0 | 画像前処理・リサイズ |
| データ拡張 | volumentations | 0.1.8 | 3Dデータ拡張 |
| 実験管理 | wandb | - | 実験トラッキング |
| 実験管理 | tensorboard | - | 学習曲線可視化 |

---

## 学習方法

### コマンド

```bash
cd TSwinUnet

python train.py \
    --exp_root 'チェックポイント保存パス' \
    --config_file './configs/tswin_unet/exp3.yaml' \
    --train-df "学習データリストCSVのパス" \
    --data_root "学習データのルートパス"
```

### オプション引数

| 引数 | 説明 | デフォルト |
|------|------|--------|
| `--exp_root` | チェックポイント保存先ディレクトリ | 必須 |
| `--config_file` | 設定YAMLファイルのパス | 必須 |
| `--train-df` | 学習データリストCSVのパス | 必須 |
| `--data_root` | 学習データのルートディレクトリ | 必須 |
| `--resume` | チェックポイントから再開（フラグ） | False |
| `--norm_label` | ラベルを正規化（フラグ） | False |

### 学習処理の流れ（`trainer.py`）

1. **データ分割**：CSVリストをランダム80/20に分割（学習/検証）
2. **エポックループ**（最大150エポック）：
   - `_train_epoch()`：Mixed Precision（FP16）で順伝播・逆伝播
   - `_val_epoch()`：検証データで RMSE を計算
3. **チェックポイント保存**：検証RMSEが改善した場合に自動保存
4. **早期終了**：20エポック改善なしで自動停止
5. **学習率スケジューラ**：WarmupCosineAnnealing（10エポックウォームアップ後、コサインアニーリング）

---

## 推論方法

### コマンド

```bash
cd TSwinUnet

python predict.py \
    --config_file './configs/tswin_unet/exp3.yaml' \
    --output_root '予測結果出力パス' \
    --exp_root 'チェックポイントパス' \
    --test-df "テストデータリストCSVのパス" \
    --data_root "テストデータのルートパス"
```

### 出力形式
- GeoTIFF（`.tif`）形式で建物高さマップを出力
- 元の衛星データと同じ地理参照情報（CRS・ジオトランスフォーム）を保持

---

## 設定ファイル解説（`configs/tswin_unet/exp3.yaml`）

```yaml
exp:  exp3        # 実験名
seed: 42          # 再現性確保のためのランダムシード
cv:   5           # 交差検証のFold数（実際には使用しない）

loader:
  train_batch:   4      # 学習バッチサイズ（GPU VRAM次第で調整）
  val_batch:     1      # 検証バッチサイズ
  num_workers:   16     # データ読み込み並列ワーカー数
  pin_memory:    true   # ピンドメモリ（GPU転送高速化）
  apply_augment: true   # データ拡張の有効化
  s1_index_list: all    # 使用するSAR帯域（all=全4バンド）
  s2_index_list: all    # 使用する光学帯域（all=全11バンド）
  months_list: [0..11]  # 使用する月（0=1月、11=12月）

trainer:
  epochs:     150   # 最大学習エポック数
  accum_iter: 1     # 勾配累積ステップ数
  ckpt_freq:  5000  # チェックポイント保存頻度（ステップ数）
  early_stop: 20    # 早期終了の閾値（エポック数）

loss:
  rec:  {mode: rmse_nonzero, weight: 2.0}  # 高さ回帰損失（非ゼロピクセルのRMSE）
  iou:  {mode: iou,          weight: 1.0}  # IoU損失（セグメンテーション整合性）
  rseg: {mode: DICE,         weight: 1.0}  # Dice損失（逆セグメンテーション）
  seg:  {mode: DICE,         weight: 1.0}  # Dice損失（セグメンテーション）

optimizer:
  mode:         adamw    # AdamWオプティマイザ
  lr:           0.0001   # 初期学習率
  betas:        [0.9, 0.99]
  weight_decay: 0.1      # L2正則化係数

scheduler:
  min_lr: 0.000001  # 最小学習率
  warmup: 10        # ウォームアップエポック数

model:
  name: tswin_unet
  params:
    input_dim:        9    # 入力チャンネル数（S1:4 + S2:5）
    encoder_backbone: tf_efficientnet_b5_ns  # EfficientNet-B5エンコーダ
    n_head:           8    # 時間的注意のヘッド数
    d_model:          256  # 注意機構の埋め込み次元
    patch_size:       [1, 2, 2]  # Swin Transformerパッチサイズ
    window_size:      [3, 7, 7]  # Swinアテンションウィンドウサイズ
    ts_channels:      12   # 時系列チャンネル数（月数）
    drop_path_rate:   0.1  # Stochastic Depth正則化
    attn_version:     v2   # SwinTransformer v2
```

---

## 損失関数の詳細（`libs/utils/losses.py`）

学習では4種類の損失関数を組み合わせて使用します：

| 損失関数 | 重み | 目的 |
|---------|------|------|
| **RMSE_nonzero** | 2.0 | 建物が存在するピクセルの高さ予測精度向上 |
| **IoU Loss** | 1.0 | 2つのセグメンテーション出力の整合性 |
| **Dice Loss（rseg）** | 1.0 | 逆セグメンテーション精度向上 |
| **Dice Loss（seg）** | 1.0 | セグメンテーション精度向上 |

合計損失：`L_total = 2.0×RMSE + 1.0×IoU + 1.0×Dice(rseg) + 1.0×Dice(seg)`

---

## 評価指標

バリデーション時に以下の複合スコアで評価します：

```python
rmse = sqrt(mean((pred - label)²))          # 全ピクセルのRMSE
rmse_nonzero = sqrt(mean((pred[nz] - label[nz])²))  # 建物ピクセルのみRMSE
iou1 = IoU(seg, target_seg)                 # セグメンテーションIoU
iou2 = IoU(rseg, target_seg)                # 逆セグメンテーションIoU

val_score = rmse + rmse_nonzero + 0.5×iou1 + 0.5×iou2
```

---

## 技術的ポイント

### 時間的注意機構（LTAE2d）
12ヶ月の時系列データから、建物高さ推定に最も有益な月を自動選択する注意重みを学習します。例えば、冬季（積雪の影響が少ない月）や夏季（植生が影響する月）を適切に重み付けします。

### Mixed Precision学習
`torch.cuda.amp`（FP16自動混合精度）を使用してGPUメモリ使用量を削減し、学習速度を向上させます。

### 3Dデータ拡張
`volumentations`ライブラリによる3D空間（時間・高さ・幅）でのフリップ拡張を実装しています。

### 欠損データ処理（マスキング）
特定の月のデータが欠損している場合、確率的にランダムマスクを適用する拡張も実装しています（学習時のみ、確率50%）。

---

## 論文情報

```bibtex
@article{yadav2025high,
  title={How high are we? Large-scale building height estimation at 10 m using Sentinel-1 SAR and Sentinel-2 MSI time series},
  author={Yadav, Ritu and Nascetti, Andrea and Ban, Yifang},
  journal={Remote Sensing of Environment},
  volume={318},
  pages={114556},
  year={2025},
  publisher={Elsevier}
}
```

関連発表：
- [EGU 2024](https://meetingorganizer.copernicus.org/EGU24/EGU24-4493.html)
- [ESA URBIS 2024](https://urbis24.esa.int/urbis24-agenda/index9f7c.html?page=browseSessions&form_session=71&presentations=hide)

---

## 連絡先

**Ritu Yadav**  
📧 er.ritu92@gmail.com
