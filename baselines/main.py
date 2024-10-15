from dataset import *
from args import *
from info import *
from copy import deepcopy
import random
from evaluate import evalmodel
import time
#导入所有模型。注意下面这个灰色其实是有用的，并不是没有使用，被getattr隐含使用了而已。
# import models
# from models.JTPP.model import jtpp
# from models.SEMNPP.model import semnpp
# from models.UNIPoint.model import unipoint
# from models.THP.model import thp
# from models.SAHP.model import sahp #这里发生了一个很尴尬的事情，那就是这里上面导入了thp那个类，竟然也会运行thp所在的python文件，这导致encoder文件被导入，
# from models.EFullyNN.model import efullynn#再然后上面的sahp导入的时候encoder就会被认为是重复导入，然后就不导入sahp的encoder，到时候运行sahp的时候会变成运行thp中的encoder，真是莫名奇妙。
# from models.FullyNN.model import fullynn
# from models.RMTPP.model import rmtpp
# from models.NHP.model import nhp


# Config
def set_random_seed(seed=42):
    torch.manual_seed(seed)#torch的cpu随机性
    torch.cuda.manual_seed_all(seed)#torch的gpu随机性
    torch.backends.cudnn.benchmark = False#保证gpu每次都选择相同的算法，但是不保证该算法是deterministic的。
    torch.backends.cudnn.deterministic = True#紧接着上面，保证算法是deterministic的。
    np.random.seed(seed)#np的随机性。
    random.seed(seed)#python的随机性。
    os.environ['PYTHONHASHSEED'] = str(seed)#设置python哈希种子


def evaluation(model, dl_train, dl_val, dl_test,args):
    model.eval()

    # All training & testing sequences stacked into a single batch
    with torch.no_grad():
        # print('TRAIN')
        # _= aggregate_loss_over_dataloader(model, dl_train, eval_mode=True,logger=logger)
        # print('-'*30)
        # print('VAL')
        # _ = aggregate_loss_over_dataloader(model, dl_val, eval_mode=True,logger=logger)
        print('-'*30)
        print('TEST')
        _ = evalmodel(model, dl_test, eval_mode=True,args=args)

def train_helper(model, dl_train, dl_val,args):
    # Training config
    regularization = args.regularization  # L2 regularization parameter
    device=args.device
    learning_rate = args.lr  # Learning rate for Adam optimizer
    max_epochs = args.max_epochs  # For how many epochs to train
    display_step = args.display_step  # 原5，我们改成1       # Display training statistics after every display_step
    patience = args.patience  # 原50，我们改成3          # After how many consecutive epochs without improvement of val loss to stop training

    opt = torch.optim.Adam(model.parameters(), weight_decay=regularization, lr=learning_rate)#参数减少。

    impatient = 0
    best_loss = np.inf#越小越好，所以这里设置为无穷大。
    best_model = deepcopy(model.state_dict())#他这里不是像我那样，每当有一个更好的模型就save，而是保存一个内存变量，最后再存。
    times=[]
    for epoch in range(max_epochs):#有一说一，恕我直言，他这个好像1个epoch就差不多了，后面一直训练，好像也没有什么太大的loss改进。
        model.train()
        training_losses = []
        total_count = 0
        startt=time.time()
        for batch in dl_train:
            opt.zero_grad()#
            if batch.inter_times.shape[1]<=1:
                continue#说明一个序列只有一个事件，那么就没有必要进行预测了。这种东西按理应该先处理掉的。
            batch.inter_times = batch.inter_times.to(device)
            batch.masks = batch.masks.to(device)
            batch.marks = batch.marks.to(device)
            tot_ll,time_pred,type_pred= model.log_prob(batch)#tot_nll，就是他们通常所说的那个nll，形状是[bsize]，下面求loss，mean了一下。
            loss = -tot_ll.mean()#
            training_losses.append(-tot_ll.sum().item())#所有序列的损失和。
            loss.backward()
            opt.step()
            total_count+=len(batch.inter_times)
        endt=time.time()
        times.append(endt-startt)
        if torch.isnan(loss):#还是需要防止一下的，他妈的。
            #如果是nan，那么应该立即结束。
            print(f'Breaking due to nan at epoch {epoch}')
            break
        #否则才需要进行下面的评估，以及保存最佳模型之类的。
        model.eval()
        with torch.no_grad():
            loss_val = evalmodel(model, dl_val,args=args)#在验证集上查看一下nll.
        if (best_loss - loss_val) < 1e-4:#这个说明最佳损失更小，当前损失更大，说明训练无效。
            impatient += 1
        else:
            impatient = 0
            best_loss = loss_val
            best_model = deepcopy(model.state_dict())#保存。

        if impatient >= patience:
            print(f'Breaking due to early stopping at epoch {epoch}')
            break

        if epoch % display_step == 0:
            # print(f"Epoch {epoch:4d}: loss_train_last_batch = {loss.item():.1f}, loss_val = {loss_val:.1f}, , p_e = {impatient}")
            args.logger.info(f"Epoch {epoch:4d}: train_loss = {np.sum(training_losses)/total_count:.1f}, loss_val = {loss_val:.1f}, , p_e = {impatient}")
    args.logger.info("training speed: {} seconds per epoch".format(int(np.mean(times))))
    return best_model


def train_dataset(args):
    dataset, d_train, d_val, d_test, dl_train, dl_val, dl_test = load_data(args)#导入数据，终于来了。
    print('-'*50)
    print('Building model..')#别问我为什么下面不是使用eval，而是使用exec，我就是发现下面这样有用，但是eval会报错，搞不懂。exec好像偏向于执行命令。
    exec("from models.{}.model import {}".format(args.baseline,args.baseline.lower()))#这样不知道有没有用，只导入这一个包。
    model = eval(args.baseline.lower())(args)#最后这两个才是输入的参数。所以这里使用了一次高级用法。
    model=model.to(args.device)
    para_count=sum(p.numel() for p in model.parameters())
    args.logger.info("number of parameters:{}".format(para_count))
    print('-'*50)
    print('Training..')
    best_model = train_helper(model, dl_train, dl_val,args)#这个其实就是训练。我怎么发现这个作者好像什么也没有做，作者lnm原来就是这么写的，不过好像没有condition上？
    model.load_state_dict(best_model)
    print('-'*50)
    print('Evaluation..')
    evaluation(model, dl_train, dl_val, dl_test,args)#在所有数据集上进行评估。我发现，这个评估，在三个数据集上的结果都差不多。
    # print('-'*50)
    # print('Sampling..')#我们没法sampling，这个实在没有办法。
    # t_end,num_seq=params["t_end"],params["num_seq"]
    # pro_path=params["pro_path"]
    # sampling_plot(model, t_end, num_seq, dataset,pro_path,params['dataset_name'])#这个可以看看其实，到时候我也画，不过不太好，我这个采样真的很不方便。
    # print('-'*50)我们不再保存模型了，因为发现好像没有什么用。
    # print('Saving model ..')
    # pro_path=args.pro_path
    # model_save_path = pro_path+'/saved_models/{}.pth'.format(args.data)
    # print(model_save_path)#存储模型，没哟与必要啊。
    # torch.save(model, model_save_path)
    # torch.save(model.state_dict(), model_save_path)#

args=load_args()
logger=get_logger(args)
args.logger=logger
seed = args.seed
if args.local:
    np.random.seed(seed)
    torch.manual_seed(seed)
else:
    set_random_seed(seed)
    # torch.set_default_tensor_type(torch.cuda.FloatTensor)


#下面这个是曾经最常使用的，但是现在我不使用了，主要是有可能会占用别的gpu，例如0号。
# device=torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() and args.gpu>=0 else "cpu")
if args.local:
    device = torch.device("cpu")
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)#定义了这个之后，该gpu的编号就会对应为0，所以应该设置去使用0号设备。
    device=torch.device("cuda:0")
# if torch.cuda.is_available() and args.gpu>=0:#这里存在的一个坑，那就是一旦调用cuda.is_available()，就意味着gpu程序开始了，此时下面再去设置就没有用了，必须一开始就设置。
#     os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)#他的一个奇葩之处是定义了这个之后，该gpu的编号就会对应为0，所以应该如下这么写。
#     device=torch.device("cuda:0")
# else:
#     device=torch.device("cpu")
args.device=device

train_dataset(args)#下面那个是以字节为单位，所以除以这么多换算一下。
args.logger.info("GPU max_memory_allocated:{:.2f}GB".format(torch.cuda.max_memory_allocated(args.device)/np.power(2,30)))
args.logger.info("GPU max_memory_reserved:{:.2f}GB".format(torch.cuda.max_memory_reserved(args.device)/np.power(2,30)))
