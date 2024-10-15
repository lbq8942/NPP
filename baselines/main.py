from dataset import *
from args import *
from info import *
from copy import deepcopy
import random
from evaluate import evalmodel
import time














def set_random_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def evaluation(model, dl_train, dl_val, dl_test,args):
    model.eval()

    
    with torch.no_grad():
        
        
        
        
        
        print('-'*30)
        print('TEST')
        _ = evalmodel(model, dl_test, eval_mode=True,args=args)

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
    times=[]
    for epoch in range(max_epochs):
        model.train()
        training_losses = []
        total_count = 0
        startt=time.time()
        for batch in dl_train:
            opt.zero_grad()
            if batch.inter_times.shape[1]<=1:
                continue
            batch.inter_times = batch.inter_times.to(device)
            batch.masks = batch.masks.to(device)
            batch.marks = batch.marks.to(device)
            tot_ll,time_pred,type_pred= model.log_prob(batch)
            loss = -tot_ll.mean()
            training_losses.append(-tot_ll.sum().item())
            loss.backward()
            opt.step()
            total_count+=len(batch.inter_times)
        endt=time.time()
        times.append(endt-startt)
        if torch.isnan(loss):
            
            print(f'Breaking due to nan at epoch {epoch}')
            break
        
        model.eval()
        with torch.no_grad():
            loss_val = evalmodel(model, dl_val,args=args)
        if (best_loss - loss_val) < 1e-4:
            impatient += 1
        else:
            impatient = 0
            best_loss = loss_val
            best_model = deepcopy(model.state_dict())

        if impatient >= patience:
            print(f'Breaking due to early stopping at epoch {epoch}')
            break

        if epoch % display_step == 0:
            
            args.logger.info(f"Epoch {epoch:4d}: train_loss = {np.sum(training_losses)/total_count:.1f}, loss_val = {loss_val:.1f}, , p_e = {impatient}")
    args.logger.info("training speed: {} seconds per epoch".format(int(np.mean(times))))
    return best_model


def train_dataset(args):
    dataset, d_train, d_val, d_test, dl_train, dl_val, dl_test = load_data(args)
    print('-'*50)
    print('Building model..')
    exec("from models.{}.model import {}".format(args.baseline,args.baseline.lower()))
    model = eval(args.baseline.lower())(args)
    model=model.to(args.device)
    para_count=sum(p.numel() for p in model.parameters())
    args.logger.info("number of parameters:{}".format(para_count))
    print('-'*50)
    print('Training..')
    best_model = train_helper(model, dl_train, dl_val,args)
    model.load_state_dict(best_model)
    print('-'*50)
    print('Evaluation..')
    evaluation(model, dl_train, dl_val, dl_test,args)
    
    
    
    
    
    
    
    
    
    
    
    

args=load_args()
logger=get_logger(args)
args.logger=logger
seed = args.seed
if args.local:
    np.random.seed(seed)
    torch.manual_seed(seed)
else:
    set_random_seed(seed)
    




if args.local:
    device = torch.device("cpu")
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device=torch.device("cuda:0")





args.device=device

train_dataset(args)
args.logger.info("GPU max_memory_allocated:{:.2f}GB".format(torch.cuda.max_memory_allocated(args.device)/np.power(2,30)))
args.logger.info("GPU max_memory_reserved:{:.2f}GB".format(torch.cuda.max_memory_reserved(args.device)/np.power(2,30)))
