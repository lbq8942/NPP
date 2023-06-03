import dpp
import matplotlib.pyplot as plt
import numpy as np
import torch
from copy import deepcopy
from evaluate import eval, sampling_plot

def get_flowindex(times,args):
    
    flowindex=(times/args.windowlen).type(torch.int64)
    flowindex=torch.where(flowindex<args.flownum,flowindex,args.flownum-1)
    return flowindex



def build_model(d_train, args):
    context_size = args.hdim
    mark_embedding_size = args.mdim
    rnn_type =args.rnn_type
    device=args.device
    mean_inter_time, std_inter_time = d_train.get_inter_time_statistics()
    if args.flownum==1:
        args.windowlen=1e100
        args.section=[0,1e100]
    else:
        args.windowlen=(mean_inter_time+std_inter_time*2)/(args.flownum-1)
        args.section=list(np.arange(0,mean_inter_time+std_inter_time*2+args.windowlen/2,args.windowlen))+[1e100]
    model = dpp.models.TPPCDF(
        num_marks=d_train.num_marks,
        mean_log_inter_time=mean_inter_time,
        std_log_inter_time=std_inter_time,
        context_size=context_size,
        mark_embedding_size=mark_embedding_size,
        rnn_type=rnn_type,
        flownum=args.flownum,
        flowlen=args.flowlen,
        args=args
    )  

    return model


def evaluation(model, dl_train, dl_val, dl_test,args):
    model.eval()

    
    with torch.no_grad():
        
        
        
        
        
        print('-'*30)
        print('TEST')
        _ = eval(model, dl_test, eval_mode=True,args=args)




def sampling(params,logger,args):
    print(params['dataset_name'])
    print('-'*50)
    print('Loading data..')
    dataset, d_train, d_val, d_test, dl_train, dl_val, dl_test = load_data(params)
    print('-'*50)
    print('Building model..')
    model = build_model(d_train, params)
    pro_path=params["pro_path"]
    model_save_path = pro_path+'/models/{}-{}.pth'.format(params['dataset_name'].split('/')[-1],args.model)
    
    if args.local:
        model.load_state_dict(torch.load(model_save_path,map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(model_save_path))
    print('-' * 50)
    print('Sampling..')  
    t_end, num_seq = params["t_end"], params["num_seq"]
    pro_path = params["pro_path"]
    sampling_plot(model, t_end, num_seq, dataset, pro_path,
                  params['dataset_name'])  
    
    
    
    


































