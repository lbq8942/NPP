
from metrics import *


def evalmodel(model, dl, eval_mode=False,args=None):
    device=args.device
    if eval_mode:
        total_loss = 0.0
        total_count = 0
        
        total_nll = []
        lengths = []
        mark_preds = []
        mark_gt = []
        time_preds=[]
        time_gt=[]
        
        
        with torch.no_grad():
            for batch in dl:
                batch.inter_times = batch.inter_times.to(device)
                batch.masks = batch.masks.to(device)
                batch.marks = batch.marks.to(device)
                tot_nll ,time_pred,type_pred= model.log_prob(batch,test=True)
                total_loss += (-1)*tot_nll.sum().item()
                total_count += len(batch.inter_times)

                total_nll.append(-tot_nll.detach().cpu().numpy()) 
                 
                lengths.append(batch.masks.sum(-1).int()-1)
                mark_preds.append(type_pred)
                mark_gt.append(batch.marks[:,1:])
                time_preds.append(time_pred)
                time_gt.append(batch.inter_times[:,1:])
                
                

            total_nll = np.concatenate(total_nll)
            tot_nll = total_nll.sum() / total_count
            args.logger.info(f"NLL            : {tot_nll:.4f}")
            mark_metrics(lengths, mark_preds, mark_gt,args)
            time_metrics(lengths, time_preds, time_gt,args)
            
            
            
            
            


    else:
        total_loss = 0.0
        total_count = 0
        with torch.no_grad():
            for batch in dl:
                batch.inter_times = batch.inter_times.to(device)
                batch.masks = batch.masks.to(device)
                batch.marks = batch.marks.to(device)
                tot_nll,time_pred, type_pred= model.log_prob(batch)
                total_loss += (-1)*tot_nll.sum().item()
                total_count += len(batch.inter_times)
                
    return total_loss / total_count

