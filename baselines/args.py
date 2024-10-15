import argparse
import os


def load_args():
    parser = argparse.ArgumentParser('baseline')
    parser.add_argument('--baseline', type=str,default="PNPP", help='use which baseline', choices=["SEMNPP",
  "PNPP","JTPP","THP","SAHP","UNIPoint","EFullyNN","FullyNN","RMTPP","NHP","CDFNPP","DMTPP","SSMTPP"])
    parser.add_argument('--data', type=str, help='use which dataset')
    parser.add_argument('--proportion', type=float, help='use how many sequences',default=1)
    parser.add_argument('--datadir', type=str,default="PNPP")
    parser.add_argument('--gpu',type=int, default=0)
    parser.add_argument('--regularization', type=float,default=1e-5)
    parser.add_argument('--lr',type=float, default=0.001)
    parser.add_argument('--max_epochs',type=int, default=1000)
    parser.add_argument('--display_step', type=int,default=1)
    parser.add_argument('--patience',type=int, default=10)
    parser.add_argument('--seed', type=int,default=0, help='random seed')
    parser.add_argument('--local', action="store_true", help='use local machine or remote machine')
    parser.add_argument('--testing', action="store_true", help='training or testing')
    parser.add_argument('--load_path', type=str, default="1",help="the path of model  when training is false")
    parser.add_argument('--bs', type=int,default=64)
    parser.add_argument('--cover', type=int,default=3)
    parser.add_argument('--scale', type=float,default=1.0)
    parser.add_argument('--monum', type=int,default=20)
    parser.add_argument('--minpositive', type=float,default=1e-10)

    
    parser.add_argument('--hdim', type=int,default=64)
    parser.add_argument('--mdim',type=int, default=32)


    parser.add_argument('--baseargs', type=str,help='arguments for baseline')


    args = parser.parse_args()

    
    if args.local:
      args.pro_path = "D:/lbq/project/python/TPP"
    else:
      args.pro_path="/home/liubingqing/project/TPP"
    args.data_path = args.pro_path+"/data"


    return args