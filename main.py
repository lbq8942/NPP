import numpy as np
from copy import deepcopy
from dataset import *
from args import *
from log import *
from train import *
import random

def set_random_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)



def train_helper(model, dl_train, dl_val,args):
    
    regularization = args.regularization  
    device=args.device
    learning_rate = args.lr  
    max_epochs = args.max_epochs  
    display_step = args.display_step  
    patience = args.patience  

    opt = torch.optim.Adam(model.parameters(), weight_decay=regularization, lr=learning_rate)

    impatient = 0
    best_loss = np.inf
    best_model = deepcopy(model.state_dict())
    for epoch in range(max_epochs):
        model.train()
        training_losses = []
        total_count = 0
        for batch in dl_train:
            opt.zero_grad()
            batch.inter_times = batch.inter_times.to(device)
            batch.masks = batch.masks.to(device)
            batch.marks = batch.marks.to(device)
            
            batch.flowindex=get_flowindex(batch.inter_times,args)
            tot_nll, mark_class ,time_pred,_,_= model.log_prob(batch)
            loss = -tot_nll.mean()
            training_losses.append(-tot_nll.sum().item())
            loss.backward()
            opt.step()
            total_count+=len(batch.inter_times)

        model.eval()
        with torch.no_grad():
            loss_val = eval(model, dl_val,args=args)
        if (best_loss - loss_val) < 1e-4:
            impatient += 1
            if loss_val < best_loss:
                best_loss = loss_val
                best_model = deepcopy(model.state_dict())
        else:
            best_loss = loss_val
            best_model = deepcopy(model.state_dict())
            impatient = 0

        if impatient >= patience:
            print(f'Breaking due to early stopping at epoch {epoch}')
            break

        if epoch % display_step == 0:
            
            args.logger.info(f"Epoch {epoch:4d}: train_loss = {np.sum(training_losses)/total_count:.1f}, loss_val = {loss_val:.1f}, , p_e = {impatient}")

    return best_model


def train_dataset(args):
    dataset, d_train, d_val, d_test, dl_train, dl_val, dl_test = load_data(args)
    print('-'*50)
    print('Building model..')
    model = build_model(d_train, args)
    model=model.to(args.device)
    print('-'*50)
    print('Training..')
    best_model = train_helper(model, dl_train, dl_val,args)
    model.load_state_dict(best_model)
    print('-'*50)
    print('Evaluation..')
    evaluation(model, dl_train, dl_val, dl_test,args)
    print('-'*50)
    
    
    
    
    print('-'*50)
    print('Saving model ..')
    pro_path=args.pro_path
    model_save_path = pro_path+'/saved_models/{}.pth'.format(args.data)
    print(model_save_path)
    
    torch.save(model.state_dict(), model_save_path)



args=load_args()
logger=get_logger(args)
args.logger=logger
args.logger.info(args)
seed = args.seed
if args.local:
    np.random.seed(seed)
    torch.manual_seed(seed)
else:
    set_random_seed(seed)
    

args.rnn_type = "GRU"   

device=torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() and args.gpu>=0 else "cpu")
args.device=device
if args.testing:
    if args.model==0:
        args.device="cpu"
    
    eval_dataset(args)
else:
    train_dataset(args)

