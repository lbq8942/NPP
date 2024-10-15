import argparse
import os
#这个是超级无敌终极版，将集合所有的baseline。

def load_args():
    parser = argparse.ArgumentParser('baseline')
    parser.add_argument('--baseline', type=str,default="PNPP", help='use which baseline', choices=["SEMNPP",
  "PNPP","JTPP","THP","SAHP","UNIPoint","EFullyNN","FullyNN","RMTPP","NHP","CDFNPP","DMTPP","SSMTPP"])
    parser.add_argument('--data', type=str, help='use which dataset')
    parser.add_argument('--proportion', type=float, help='use how many sequences',default=1)#使用多少数量的数据集，感觉这个东西是以后必须要学会的，调试太重要了，否则一直出错。
    parser.add_argument('--datadir', type=str,default="PNPP")#datapath:data_path+datadir+data
    parser.add_argument('--gpu',type=int, default=0)#多少维的向量表示一个事件类型。
    parser.add_argument('--regularization', type=float,default=1e-5)#多少维的向量表示一个历史。
    parser.add_argument('--lr',type=float, default=0.001)#多少维的向量表示一个事件类型。
    parser.add_argument('--max_epochs',type=int, default=1000)#多少维的向量表示一个事件类型。
    parser.add_argument('--display_step', type=int,default=1)#多少维的向量表示一个历史。
    parser.add_argument('--patience',type=int, default=10)#多少维的向量表示一个事件类型。
    parser.add_argument('--seed', type=int,default=0, help='random seed')
    parser.add_argument('--local', action="store_true", help='use local machine or remote machine')
    parser.add_argument('--testing', action="store_true", help='training or testing')#这个布尔值非常魔幻，--training False不行，会被当成布尔值，从而是True。
    parser.add_argument('--load_path', type=str, default="1",help="the path of model  when training is false")#如果是在测试的时候，该导入哪一个模型。
    parser.add_argument('--bs', type=int,default=64)
    parser.add_argument('--cover', type=int,default=3)#3delta法则确定我们主要建模的空间，这个用来分段函数，时间预测，那个期望不是0，无穷，而是这个法则。
    parser.add_argument('--scale', type=float,default=1.0)#正常是不需要scale的。这个scale会发生在输入到分布，求解密度的时候。
    parser.add_argument('--monum', type=int,default=20)#所有蒙特卡洛求积分的时候的采样数量。
    parser.add_argument('--minpositive', type=float,default=1e-10)#所有会报错的时候，我们使用这个数字来防止其为0.

    #下面这两个讲道理非常特殊，按理应该是模型相关参数，但是点过程里面普遍是有这两个参数的，所以放到这里也没有毛病。
    parser.add_argument('--hdim', type=int,default=64)#多少维的向量表示一个历史。
    parser.add_argument('--mdim',type=int, default=32)#多少维的向量表示一个事件类型。


    parser.add_argument('--baseargs', type=str,help='arguments for baseline')#默认就是None


    args = parser.parse_args()

    # args.pro_path="/data/liubingqing/debug/TPPBASE/TPP"
    if args.local:
      args.pro_path = "D:/lbq/project/python/TPP"
    else:
      args.pro_path="/home/liubingqing/project/TPP"
    args.data_path = args.pro_path+"/data"


    return args