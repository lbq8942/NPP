
from metrics import *


def evalmodel(model, dl, eval_mode=False,args=None):#牛批，这个竟然是dpp内置的。不对，原来的作者并没有这个，而是新作者加上去的。
    device=args.device
    if eval_mode:#这个是比较详细的评估，专门用于训练好了模型之后，才会动用这个选项，否则是进入else选项。
        total_loss = 0.0
        total_count = 0
        
        total_nll = []
        lengths = []
        mark_preds = []
        mark_gt = []
        time_preds=[]
        time_gt=[]
        # Fs=[]
        # samples=[]
        with torch.no_grad():
            for batch in dl:
                batch.inter_times = batch.inter_times.to(device)
                batch.masks = batch.masks.to(device)
                batch.marks = batch.marks.to(device)
                tot_nll ,time_pred,type_pred= model.log_prob(batch,test=True)
                total_loss += (-1)*tot_nll.sum().item()#[bsize]其他的都是[bsize,seqlen]，不对,log_surv也是[bsize]
                total_count += len(batch.inter_times)

                total_nll.append(-tot_nll.detach().cpu().numpy()) #(batch_size,)
                 
                lengths.append(batch.masks.sum(-1).int()-1)#[bs,seqlen]减去1是因为第一个事件不进行预测。
                mark_preds.append(type_pred)#
                mark_gt.append(batch.marks[:,1:])#不要第一个的意思
                time_preds.append(time_pred)#[bs,seqlen]而且在填充的地方都填充了0好像。上面有一个lengths的记录，这使得似乎不需要填充0就可以。
                time_gt.append(batch.inter_times[:,1:])#不要第一个的意思
                # Fs.append(F)
                # samples.append(sample)

            total_nll = np.concatenate(total_nll)
            tot_nll = total_nll.sum() / total_count
            args.logger.info(f"NLL            : {tot_nll:.4f}")
            mark_metrics(lengths, mark_preds, mark_gt,args)
            time_metrics(lengths, time_preds, time_gt,args)#这里把那个中值time_q2s给放进去就可以显示根据真实模型的时间预测结果了，
            #不过我试过，根据中值反而比我们的差，原因很简单，它带入interval的时候都不是0.5，反而是0.46啥的，反而我们的更接近，比如0.48，0.9之类的。
            #个人猜测可能真实模型使用期望计算会更准，但是我这里就不去计算了，计算了这个的话，那众数也应该计算，pdf的众数老实讲有点难计算。而且计算了这个，那
            #你的那个nll是不是也可以计算一下，从而都得到一个天花板，突然好多工作，所以暂时就不搞这个东西了。
            # F_metrics(lengths, Fs,args)#这里把那个中值time_q2s给放进去就可以显示根据真实模型的时间预测结果了，
            # sample_metrics(lengths,time_gt,mark_gt,samples,args)


    else:#奇怪，
        total_loss = 0.0
        total_count = 0
        with torch.no_grad():#不要梯度，这个是完全没有问题的，但是我们那个就有问题了。
            for batch in dl:
                batch.inter_times = batch.inter_times.to(device)
                batch.masks = batch.masks.to(device)
                batch.marks = batch.marks.to(device)
                tot_nll,time_pred, type_pred= model.log_prob(batch)#还是一毛一样。
                total_loss += (-1)*tot_nll.sum().item()#但是我们发现，其只需要tot_nll，比较这个即可。[bsize]sum,-1那么就是越大越好了。
                total_count += len(batch.inter_times)#一共有多少个序列。
                
    return total_loss / total_count

