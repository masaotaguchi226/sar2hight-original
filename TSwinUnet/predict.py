import yaml
import pickle
import argparse
import warnings
from libs.utils import *
from libs.predict import *
from omegaconf import OmegaConf
from os.path import join as opj
from sklearn.model_selection import KFold, train_test_split
warnings.filterwarnings('ignore')

def main(args):
    # --------------------------------------------------------------------------
    # 設定ファイルの読み込み
    with open(args.config_file, 'r') as f:
        configs = yaml.load(f, Loader=yaml.Loader)
    configs = OmegaConf.create(configs)

    # データ分割の読み込み
    df: pd.DataFrame = pd.read_csv(args.test_df)
        
    if args.val:
        train_df, dev_df = train_test_split(df, test_size=0.2, random_state=42)
        val_list = dev_df['fname'].tolist()
    
    else:
        val_list = df['fname'].tolist()
    print('テストデータ数', len(val_list))

    # --------------------------------------------------------------------------
    # テストデータの推論
    exp_dir = opj(args.exp_root)
    data_dir = args.data_root
    output_dir = opj(args.output_root)

    print('-' * 100)
    print('推論中 ...\n')
    print(f'- データディレクトリ  : {data_dir}')
    print(f'- 実験ディレクトリ   : {exp_dir}')
    print(f'- 出力ディレクトリ   : {output_dir}')
    print(f'- 設定内容   : {args.config_file}')

    model_paths = opj(exp_dir, 'model.pth')
    predictor = BHEPredictor(
        model_path    = model_paths,
        configs        = configs,
    )
    predictor.predict(data_dir, val_list, output_dir)
    
    print('-' * 100, '\n')
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='推論')
    parser.add_argument( "--test-df", type=str, default='<テストCSVリストのパス>', help="テストCSVデータリストのパス",)
    parser.add_argument('--data_root',      type=str, default='<テストデータのパス>', help='テストデータのディレクトリパス')
    parser.add_argument('--exp_root',       type=str, default='<実験チェックポイントフォルダ>', help='実験ルートディレクトリ')
    parser.add_argument('--output_root',    type=str, default='<予測結果出力フォルダ>', help='出力ルートディレクトリ')
    parser.add_argument('--config_file',    type=str, default ='<設定YAML>', help='設定のYAMLファイルパス')
    parser.add_argument('--val',     action='store_true', help='検証精度を評価する場合に指定（未指定の場合はテスト精度）')
    args = parser.parse_args()
    check_predict_args(args)
    main(args)