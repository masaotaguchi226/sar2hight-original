import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import yaml
import pickle
import argparse
import warnings
import torch
from libs.utils import *
from libs.train import *
from omegaconf import OmegaConf
from sklearn.model_selection import KFold, train_test_split
warnings.filterwarnings('ignore')

def main(args):

    # --------------------------------------------------------------------------
    # 設定ファイルの読み込み
    with open(args.config_file, 'r') as f:
        configs = yaml.load(f, Loader=yaml.Loader)
    configs = OmegaConf.create(configs)

    # --------------------------------------------------------------------------
    # データ分割の読み込み
    df: pd.DataFrame = pd.read_csv(args.train_df)
    train_df, dev_df = train_test_split(df, test_size=0.2, random_state=42)

    train_list = train_df['fname'].tolist()
    val_list = dev_df['fname'].tolist()
    print('学習データと検証データ', len(train_list), len(val_list))

    # --------------------------------------------------------------------------
    # 環境の初期化
    init_environment(configs.seed)

    print('-' * 100)
    print('学習中 ...\n')
    print(f'- データルート : {args.data_root}')
    print(f'- 実験ディレクトリ   : {args.exp_root}')
    print(f'- 設定ファイル   : {args.config_file}')
    print(f'- 設定内容   : {args.config_file}')
    print(f'- 正規化 : {args.norm_label}')
    print(f'- 学習データ数 : {len(train_list)}')
    print(f'- 検証データ数   : {len(val_list)}\n')

    loader_kwargs = dict(
        configs        = configs.loader#,
    )
    train_loader = get_dataloader('train', train_list, args.data_root, args.norm_label, **loader_kwargs)
    val_loader   = get_dataloader('val',   val_list,   args.data_root, args.norm_label, **loader_kwargs)    
    print('GPUメモリ情報', torch.cuda.mem_get_info())

    # トレーナーの初期化
    trainer = PytorchTrainer(
        configs = configs,
        exp_dir = args.exp_root,
        resume  = args.resume,
        label_norm = args.norm_label
    )

    # モデルの学習
    trainer.forward(train_loader, val_loader)    
    print('-' * 100, '\n')    
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='学習')
    parser.add_argument( "--train-df", type=str, default='<学習CSVリストのパス>', help="学習CSVデータリストのパス",)
    parser.add_argument('--data_root',      type=str, default='<学習データのパス>', help='学習データのディレクトリパス')
    parser.add_argument('--exp_root',       type=str, default='<チェックポイントのパス>', help='実験ルートディレクトリ')
    parser.add_argument('--config_file',    type=str, default ='<設定YAMLファイル>', help='設定のYAMLファイルパス')
    parser.add_argument('--resume',         action='store_true', help='チェックポイントから再開する場合に指定')
    parser.add_argument('--norm_label',     action='store_true', help='正規化ラベルを使用する場合に指定')
    args = parser.parse_args()

    check_train_args(args)
    main(args)