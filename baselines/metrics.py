from sklearn import metrics
from sklearn.metrics import classification_report
import numpy as np
import torch

def mark_metrics(lengths, mark_pred, mark_gt, args):
    mark_pr_f = []
    mark_gt_f = []
    for i in range(len(lengths)):
        for j in range(len(lengths[i])):
            for k in range(lengths[i][j]):
                mark_pr_f.append(mark_pred[i][j][k].detach().cpu())
                mark_gt_f.append(mark_gt[i][j][k].detach().cpu())

    acc = metrics.accuracy_score(mark_gt_f, mark_pr_f) * 100
    f1_micro = metrics.f1_score(mark_gt_f, mark_pr_f, average='micro') * 100
    f1_macro = metrics.f1_score(mark_gt_f, mark_pr_f, average='macro') * 100
    f1_weighted = metrics.f1_score(mark_gt_f, mark_pr_f, average='weighted') * 100

    args.logger.info(f"f1_micro (Acc) : {f1_micro:.4f}")
    args.logger.info(f"f1_macro       : {f1_macro:.4f}")
    args.logger.info(f"f1_weighted    : {f1_weighted:.4f}")
    #这个是新加的东西。
    # args.logger.info(classification_report(mark_gt_f, mark_pr_f))#暂时好像不需要了，因为并没有看出我的对事件类型少的有了很大的进步。

    


def time_performance(truth, pred):
    # 一种方法就是i计算那个mean absolute error
    mae = np.abs(truth - pred).mean()
    mse = np.power(truth - pred, 2).mean()
    return mae, mse  # 好像最经常使用的也就是这两个了。


def time_metrics(lengths, time_pred, time_gt, args):
    mus = []
    medians = []
    modes = []

    mark_gt_f = []
    for i in range(len(lengths)):
        for j in range(len(lengths[i])):
            for k in range(lengths[i][j]):
                mus.append(time_pred[i][j][k].detach().cpu())
                mark_gt_f.append(time_gt[i][j][k].detach().cpu())

    mark_gt_f = np.array(mark_gt_f)
    mus = np.array(mus)
    mae, mse = time_performance(mark_gt_f, mus)
    args.logger.info(f"Time prediction by expectation: mae,mse    : {mae:.4f},{mse:.4f}")


def F_metrics(lengths, Fs, args):
    mus = []
    for i in range(len(lengths)):  # batch数量
        for j in range(len(lengths[i])):  # 批大小
            for k in range(lengths[i][j]):  # 位置
                mus.append(Fs[i][j][k].detach().cpu())

    args.logger.info(f"F prediction: {np.mean(mus):.4f}")

def sample_metrics(lengths,time_gt,mark_gt,samples,args):
    marks=[]#每一个序列弄成一个列表。
    times=[]
    marks_sample=[]
    times_sample=[]


    for i in range(len(lengths)):
        for j in range(len(lengths[i])):
            mark = []
            time=[]
            mark_sample = []
            time_sample=[]
            leng=lengths[i][j]
            for k in range(leng//2,leng):#只要后一半。
                mark.append(mark_gt[i][j][k].detach().cpu().item())
                time.append(time_gt[i][j][k].detach().cpu().item())
                mark_sample.append(samples[i][1][j][k].detach().cpu().item())
                time_sample.append(samples[i][0][j][k].detach().cpu().item())
            marks.append(mark)
            times.append(time)
            marks_sample.append(mark_sample)
            times_sample.append(time_sample)


    #然后应该就可以比较了，但是我们只可以展示其中一部分的东西，为了展示得好一些，我们只将这些结果保存，然后留给jupyter查看。
    torch.save((marks,times,marks_sample,times_sample),args.pro_path+"/log/"+args.baseline+"/{}sample".format(args.data))
