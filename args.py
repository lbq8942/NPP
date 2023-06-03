import argparse

def load_args():
    parser = argparse.ArgumentParser('TPPCDF')

    parser.add_argument('--data', type=str,default="hawkes_ind", help='use which dataset')
    parser.add_argument('--bs', type=int,default=64)
    parser.add_argument('--flownum',type=int, default=6)
    parser.add_argument('--flowlen',type=int, default=1)
    parser.add_argument('--hdim', type=int,default=64)
    parser.add_argument('--mdim',type=int, default=32)
    parser.add_argument('--headsnum',type=int, default=1)
    parser.add_argument('--gpu',type=int, default=0)
    parser.add_argument('--regularization', type=float,default=1e-5)
    parser.add_argument('--lr',type=float, default=0.001)
    parser.add_argument('--max_epochs',type=int, default=1000)
    parser.add_argument('--display_step', type=int,default=1)
    parser.add_argument('--patience',type=int, default=10)
    parser.add_argument('--basis',type=int, default=0)
    parser.add_argument('--map',type=int, default=0)
    parser.add_argument('--seed', type=int,default=0, help='random seed')
    parser.add_argument('--local', action="store_true", help='use local machine or remote machine')
    parser.add_argument('--testing', action="store_true", help='training or testing')
    parser.add_argument('--load_path', type=str, default="1",help="the path of model  when training is false")

    args = parser.parse_args()

    if args.local:
        args.pro_path="D:\lbq\lang\pythoncode\pycharm project\TPPBASE\TPPSUM"
    else:
        args.pro_path="/data/liubingqing/debug/TPPBASE/TPPSUM"

    return args


