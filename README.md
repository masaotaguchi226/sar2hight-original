## [大規模建物高さ推定：Sentinel-1 SARとSentinel-2 MSI時系列を活用](https://www.sciencedirect.com/science/article/pii/S0034425724005820)

本研究では、Sentinel-1 SARおよびSentinel-2 多波長画像の時系列データを活用した大規模建物高さ推定のための高度な深層学習モデル **T-SwinUNet** を提案します。このモデルは、オランダ・スイス・エストニア・ドイツのデータで学習・評価され、その汎化性能は他のヨーロッパ諸国10都市からなる分布外（OOD）テストセットで検証されています。T-SwinUNetは、10m空間解像度において二乗平均平方根誤差（RMSE）1.89 mで建物高さを予測し、最先端モデルを上回る性能を示しています。OODテストセットでの高い汎化性能（RMSE 3.2 m）は、ヨーロッパ全域での低コスト建物高さ推定、および将来的な他地域への展開可能性を示しています。さらに、100m解像度での評価では、T-SwinUNet（RMSE 0.29 m、R² 0.75）がグローバル建物高さプロダクトGHSL-Built-H R2023A（RMSE 0.56 m、R² 0.37）をも上回る性能を示しました。

<img src="https://github.com/RituYadav92/Large-Scale-Building-Height-Estimation/blob/main/TSwinUnet/assets/figures/dataset_location.png" alt="サイト" width="500" height="400">

### 🎉 論文
Remote Sensing of Environment - https://www.sciencedirect.com/science/article/pii/S0034425724005820

関連発表：👉 [EGU 2024](https://meetingorganizer.copernicus.org/EGU24/EGU24-4493.html) & 
         👉 [ESA URBIS 2024](https://urbis24.esa.int/urbis24-agenda/index9f7c.html?page=browseSessions&form_session=71&presentations=hide)


### 🛠️ セットアップ
以下のコマンドでconda環境を作成します。

```bash
conda env create -f environment.yml
```

### 🏋️‍♂️ 学習
以下のように `train.py` スクリプトを実行します。

```bash
python train.py \
    --exp_root 'チェックポイント保存パス' \
    --config_file './configs/tswin_unet/exp3.yaml' \
    --train-df "学習データリストCSV" \
    --data_root "学習データパス"
```
###  🚀 推論
以下のように `predict.py` スクリプトを実行します。
```bash
python predict.py \
    --config_file './configs/tswin_unet/exp3.yaml' \
    --output_root '予測結果出力パス' \
    --exp_root 'チェックポイントパス' \
    --test-df "テストデータリストCSV" \
    --data_root "テストデータパス"
```

### 🎉 データセット
データセット：[M4Heights](https://huggingface.co/datasets/Rituxx96x/M4Heights)
このデータセットは学習で使用した実際のデータとは異なりますが、M4Heightsには本研究で使用した4か国のうち3か国分のSentinel-1・Sentinel-2時系列データおよびリファレンスが含まれています。タスクに有用なデータセットとなっています。使用前にデータセットの説明をご確認ください。詳細についてはお気軽にお問い合わせください。

### 📈 結果

<img src="https://github.com/RituYadav92/Large-Scale-Building-Height-Estimation/blob/main/TSwinUnet/assets/figures/Quant.jpg" alt="定量評価" width="900" height="145">
<img src="https://github.com/RituYadav92/Large-Scale-Building-Height-Estimation/blob/main/TSwinUnet/assets/figures/COR.jpg" alt="相関" width="680" height="350">
<img src="https://github.com/RituYadav92/Large-Scale-Building-Height-Estimation/blob/main/TSwinUnet/assets/figures/GEE_vis.jpg" alt="GEE可視化" width="900" height="450">

## 🎓 引用

本論文を引用する際は以下をご使用ください：

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

### 👋 連絡先
Ritu Yadav（メール：er.ritu92@gmail.com）
