#一个文件完成它的所有，其实关键问题在于它的参数按照道理一个args是不可能满足所有模型的，那我们怎么办呢？所以每一个模型里面都有一个args。
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as tnf
from torch.autograd import grad
#这个是超级无敌终极版，将集合所有的baseline。
model_parameters={
    "addtimelayer":1,#默认是添加时间预测层的。
    "coeftimeloss": 0.01,  # 也就是时间损失尽量小一些，主要还是搞那个
    "residual":0#是否启用残差网络，不知道为什么我看到的结果是启用残差好像反而更差一点，无语了。
}

def updata_args(args):
    #这里需要将args给说清楚。args是一个字典格式--args d:daf,的形式。
    para=args.baseargs
    arg_dict = args.__dict__
    args.__dict__.update(model_parameters)#注意一下更新顺序，先填入默认的，然后更新我们填入的。
    paradict = {}
    if para != None:
        keyvalues=para.split(",")#可以得到这么多个参数。不对啊，其实我们还应该和那个子args合并，否则是不行的，所以一并放到
        for keyvalue in keyvalues:
            key,value=keyvalue.split(":")
            arg_dict[key]=type(arg_dict[key])(value)#数据转换，这里完成更新。
    # arg_dict.update(paradict)  #更改这个字典就等价于更改args，十分方便。的确更改了，但是一个问题，数据结构的类型改变了，比如整数，浮点数之类的。
    #这里只是暂时更新而已，后面还要继续更新。似乎这里不应该更新，而应该到后面的时候再更新。
    return args


class Encoder(nn.Module):#目前只实现了这一个，讲道理还可以是transformer的。
    def __init__(self,args):
        super(Encoder,self).__init__()
        self.args=args
        self.arch = nn.GRU(input_size=args.mdim+1, hidden_size=args.hdim,batch_first=True)
        # self.context_init = nn.Parameter(torch.zeros(args.hdim))#不再需要预测第一个了。
    def forward(self,x):#下面好多都被注释了，因为很简单，不需要预测第一个了。
        #x[bs,seqlen,hdim]这个就是输入向量。我们使用的是RNN。
        context = self.arch(x)[0][:, :-1, :]#rnn返回的东西是什么？其实应该是有两部分的，一个是记忆，一个是隐状态，但是我们只要隐状态。
        # batch_size, seq_len, context_size = context.shape
        # context_init = self.context_init[None, None, :].expand(batch_size, 1, -1)  # (batch_size, 1, context_size)
        # Shift the context by vectors by 1: context embedding after event i is used to predict event i + 1
        # context = context[:, :-1, :]#[bs,seqlen-1,hdim]最后一个不需要用来预测。
        # context = torch.cat([context_init, context], dim=1)#需要预测第一个
        return context#[bs,seqlen,hdim]就是这个玩意了。

class MLP(nn.Module):#这个要慎重使用，我们使用的激活函数有点略微区别。
    def __init__(self,dims):#这个难点在于是一个多层结构。然后那个多少层就取决于dims了。
        super(MLP,self).__init__()
        lnum=len(dims)-1
        assert lnum>0
        self.para=nn.ModuleList()
        for i in range(lnum):
            self.para.append(nn.Linear(dims[i],dims[i+1]))
        self.lnum=lnum

    def forward(self,x,act=torch.relu):
        lnum=self.lnum
        for i in range(lnum):
            x=self.para[i](x)
            if (i+1)!=lnum:#如果不是最后一层。
                x=act(x)#那么就需要激活。
        return x

def x3(x):
    return torch.pow(x,3)#特点就是，是一个单调的递增函数。

class Decoder(nn.Module):#这个其实就是一个MLP，根据h生成参数.
    def __init__(self,args):
        super(Decoder,self).__init__()
        #fullynn是使用神经网络建模风险函数。f1(h,t)，关于t的所有参数都需要是正数。
        #三个东西，第一个是f1(h,t)一个是f2(h)用于预测时间，但是可选，另外一个是f3(h,t)用于预测事件类型，从而p(m,t)=p(t)p(m|t)没有毛病。
        self.hazard_nn_h1= nn.Linear(args.hdim,args.hdim)#这个不需要是正数。
        if args.residual:#如果启用残差网络。
            self.residual_nn=nn.Linear(args.hdim, 1)

        self.hazard_nn_t1 = nn.Parameter(torch.rand(args.hdim))#这个需要是非负数。

        self.hazard_nn=MLP([args.hdim, args.hdim, 1])#这个搞多少层比较好呢？我这里的话，就搞了一层，感觉会比较弱啊。所以搞了两层。
        #然后是事件类型预测层。
        self.type_nn=MLP([args.hdim+1,args.hdim,args.m])#没有毛病，也是两层。
        if args.addtimelayer:
            self.time_nn=MLP([args.hdim,args.hdim,1])#也是两层。
        self.args = args
        self.actfn=torch.tanh#要求是必须正单调，而且正无穷的地方必须也是正无穷。那我为什么还用这个函数？其实好像不需要一定是正无穷就正无穷，这只是对应在无穷远处强度函数仍然有值的情况，但是我们
        # self.actfn=nn.PReLU()#要求是必须正单调，而且正无穷的地方必须也是正无穷。那我为什么还用这个函数？其实好像不需要一定是正无穷就正无穷，这只是对应在无穷远处强度函数仍然有值的情况，但是我们

        #真实的例子是，通常过不了多久就会发生一个事件，这意味着按照常理，无穷远处强度就应该是0才对。
        # self.actfn=x3#试过了这个不行。效果非常的差。
        # self.actfn = torch.relu#这个也差，虽然没有上面那个这么离谱。

    def make_pos(self):
        self.hazard_nn_t1.data*= (self.hazard_nn_t1.data>=0)#不是module，不能迭代，直接使用data。
        for p in self.hazard_nn.parameters():
            p.data *= (p.data >= 0)  # 取绝对值。

    def get_hazard(self,h,t):#[bs,seqlen,hdim][bs,seqlen,k]
        if len(t.shape)==3:
            t=t.unsqueeze(3)#[bs,seqlen,k,1]
        else:
            t = t.unsqueeze(2).unsqueeze(3)# [bs,seqlen,1=k,1]

        h2 = self.hazard_nn_h1(h)  # [bs,seqlen,hdim]
        t2 = (self.hazard_nn_t1.unsqueeze(0).unsqueeze(0).unsqueeze(0)) *t
            # [1,1,1,hdim][bs,seqlen,k,1]=[bs,seqlen,k,hdim]

        ht2 = self.actfn(h2.unsqueeze(2) + t2)  # [bs,seqlen,1,hdim]+[bs,seqlen,k,hdim]=[bs,seqlen,k,hdim]
        hazardt = self.hazard_nn(ht2,
                                 act=self.actfn)  # [bs,seqlen,k,1]中间层需要激活，最后一层也需要激活，不对啊，
        # 最后一层不需要激活，就算是为负数也没有关系，我们本质是会让f(h,t)-f(h,0)的。所以只要保证是单调递增就行。
        if self.args.residual:
            res = self.residual_nn(h)# [bs,seqlen,1]残差加了一个alphat没有毛病。
            residual = torch.relu(res[..., [-1]].unsqueeze(2)) * t  # [bs,seqlen,1,1]*[bs,seqlen,k,1]=[bs,seqlen,k,1]
            hazardt=residual+hazardt#[bs,seqlen,k,1]这样应该就没有毛病了。可以放心地使用tanh激活函数了。
        return hazardt

    def forward(self,h,t):#[bs,seqlen,hdim][bs,seqlen,k]返回的是该点的强度值。k表示的就是我们要求解多个强度函数在多个位置的值。
        #先确保对应的参数全为正数。
        self.make_pos()
        #我们这里需要使用求导功能。
        t.requires_grad_(True)

        if self.training==False:#说明是测试的时候
            torch.set_grad_enabled(True)#我发现，这个时候，如果你写训练为真是没有用的，必须得打开计算图，他们俩是属于不同的东西。
            hazardt = self.get_hazard(h, t)
            hazard0 = self.get_hazard(h, torch.zeros_like(t))
            hazard = (hazardt - hazard0).squeeze(3)  # [bs,seqlen,k,1][bs,seqlen,k]
            intensity=grad(hazard.sum(),t)[0]#本身都是各自独立的bsize,其实就是一个数F对另外一个数t求导。
            torch.set_grad_enabled(False)#我发现，这个时候，如果你写训练为真是没有用的，必须得打开计算图，他们俩是属于不同的东西。
        else:#训练的时候：
            hazardt = self.get_hazard(h, t)
            hazard0 = self.get_hazard(h, torch.zeros_like(t))
            hazard = (hazardt - hazard0).squeeze(3)  # [bs,seqlen,k,1][bs,seqlen,k]
            intensity = grad(hazard.sum(), t, create_graph=True, retain_graph=True)[0]#这两个true确保intensity也会有梯度。
        #[bs,seqlen,k]
        #从源头杜绝intensity为0.
        intensity=intensity+self.args.minpositive
        return intensity,hazard#[bs,seqlen,k]

    def ll(self,h,t,marks):#[bs,seqlen,hdim][bs,seqlen][bs,seqlen]都是，一个是时间，一个是那个类型。
        #ll，和以前不一样了。log lambda- hazard + log p(m|t)，但是其实不太一样，为啥，之前的lambda有k，现在我们不需要了。
        timet=t*self.args.scale#不会影响nll这个东西，只有pdf的时候会影响，妈的，也会影响，要加上logs这个东西。
        intensity,hazard=self.forward(h,timet)#[bs,seqlen,1=k]->[bs,seqlen]
        hazard=hazard.squeeze(2)#[bs,seqlen]为啥intensity不需要，求导来的，自动识别了k。
        marks = tnf.one_hot(marks, num_classes=self.args.m)  # [bs,seqlen,m]
        typet=t*self.args.scale#输入到type中的t其实可以小一点的，不会影响nll这个东西。但是上面那个计算强度可能会影响。不对，好像还是不会影响，只有pdf的时候会影响。
        typelogits=torch.log_softmax(self.type_nn(torch.cat([h,typet.unsqueeze(2)],dim=-1)),dim=-1)#将h和t进行拼接。[bs,seqlen,hdim+1]->[bs,seqlen,m]
        #[bs,seqlen,m]
        m_typelogits = (typelogits * marks).sum(dim=-1)  # [bs,seqlen]这个就是取出了对应类型的概率。
        tppll=torch.log(intensity)-hazard+m_typelogits+np.log(self.args.scale)#[bs,seqlen]完工了，还有一种可能，那就是时间预测。
        ll=tppll#这个加上logs基本上是对的，有人问，那我要不要帮别人也实现一下，我觉得这个就没有必要了吧。
        if self.args.addtimelayer:
            time_pred=self.time_nn(h).squeeze(2)#[bs,seqlen,1]输出小时间。
            time_loss=-torch.pow((time_pred - typet), 2)#这个是越小越好，而目标函数要求越大越好，所以应该加上负号。
            ll+=self.args.coeftimeloss*time_loss
        return ll,tppll#[bs,seqlen]

    def tpp_prediction(self,h):#似乎我们这个不需要双重采样，因为f(t)是可以求出来的。
        # 第一层采样，tf(t)，也就是t这一层。这里应该需要一个最大时间。
        num=self.args.monum#这个可以稍微搞多那么一丢丢。我发现，如果是5，10之类的，这个 时间预测会比较不准。所以我们这里提高到了20.
        mintime,maxtime=self.args.mintime*self.args.scale,self.args.maxtime*self.args.scale
        interval=(maxtime-mintime)
        #在这个区间采样一些点。
        bs,seqlen,hdim=h.shape
        r=torch.rand(bs,seqlen,num,device=self.args.device)
        samplet=r*interval+mintime#这个就是samplet了。[bs,seqlen,num]
        #先求解出这些强度。
        intensity,hazard=self.forward(h,samplet)#[bs,seqlen,num]然后需要乘以这个samplet，然后还有一个exp。
        ft=intensity*torch.exp(-hazard)#这个就是ft了[bs,seqlen,num]
        #下面这个其实就是期望了。tf(t)(平均)*during
        time_pred=interval*((samplet*ft).mean(dim=-1))#[bs,seqlen]
        return time_pred#[bs,seqlen]

    #下面的计算分两种，看情况，如果选用了时间预测层，那么我们就直接使用这个来预测时间。
    def prediction(self,h):#这个是计算期望，感觉好难计算啊。需要两层采样，感觉好麻烦怎么办。我们这里不搞双层，想了一个更加高效的做法。可以避免双重计算。
        if self.args.addtimelayer:
            time_pred=self.time_nn(h).squeeze(2)#[bs,seqlen,1][bs,seqlen]此时会预测小的时间。
        else:
            time_pred=self.tpp_prediction(h)#[bs,seqlen]通过期望计算。这个也是预测小时间。
        type_pred=torch.argmax(self.type_nn(torch.cat([h,time_pred.unsqueeze(2)],dim=-1)), dim=-1)#[bs,seqlen]这里刚好使用的是小时间，没有毛病。
        return time_pred/self.args.scale,type_pred#最终还是还原回大时间去。

class efullynn(nn.Module):
    def __init__(
        self,
            args
    ):
        super().__init__()#这个init和原来的代码一模一样，作者没有修改。
        args=updata_args(args)
        #我们模型的专属，所以没有写到main函数里面搞成公共的。更新scale。
        if args.maxtime>5:
            args.scale=5/args.maxtime#也就是说希望最大的时间值在1左右，这个其实比较符合我们的深度网络，最好本来是-1,1的，但是我们时间必须是正数，
            #但是我怕这个效果太好，导致以后没法写新论文了，所以不要设置1，还是设置5好了。
        args.logger.info(args)
        self.num_marks = args.m
        self.args=args
        self.mean_log_inter_time = args.mean_inter_time
        self.std_log_inter_time = args.std_inter_time
        self.mark_embedding_size = args.mdim
        self.mark_embedding=nn.Embedding(args.m,args.mdim)#这里应该是要加1的，有一个填充的好像，不过好像我们填充的都是0吧。
        self.encoder=Encoder(args)
        self.decoder=Decoder(args)

    def embed(self, batch):#这个也和原来的代码一样。

        features = batch.inter_times.unsqueeze(2)#直接赋值，再也不需要log了，之前还需要log一下。  # (batch_size, seq_len, 1)
        features = (features - self.mean_log_inter_time) / self.std_log_inter_time#使用了这个东西。
        if self.num_marks > 1:
            mark_emb = self.mark_embedding(batch.marks)  # (batch_size, seq_len, mark_embedding_size)
            features = torch.cat([features, mark_emb], dim=-1)
        return features  # (batch_size, seq_len, num_features)

    def log_prob(self,batch,test=False):  # 原作者，这里得到了context之后一个linear得到了时间间隔分布的参数，用另外一个linear得到了是哪一个事件，相当于就是独立建模

        num_marks=self.num_marks
        features = self.embed(batch)  # [bsize,seqlen,hdim]
        context = self.encoder(features)  # [bsize,seqlen-1,hdim]
        ll,tppll=self.decoder.ll(context,batch.inter_times[:,1:],batch.marks[:,1:])#[bs,seqlen-1]
        ll = (ll * (batch.masks[:,1:])).sum(dim=-1)  # (batch_size, seq_len)->[bs]这样之后就可以取平均了。
        #然后接下来其实就是预测了，平常我们是不需要预测的，只有再test的情况下需要。
        if test:
            time_pred,type_pred=self.decoder.prediction(context)#都是[bs,seqlen]，这里我漏了一个东西，那就是如果先对类型进行预测该怎么办，我之前的想法是，取均匀若干个时间，然后求解概率最大值，投票，但是其实似乎均匀并不合理。
            #就这样应该就行了。
            #还有，为了公平对比，我们应该返回tppll。
            tppll = (tppll * (batch.masks[:,1:])).sum(dim=-1)#[bs]在测试的时候，使用tppll，但是其实如果不使用那个timelayer的话，两者其实是一样的。
            ll=tppll
        else:
            time_pred=0
            type_pred=0
        #这个不能为0了，需要为
        return ll,time_pred,type_pred#三要素齐全了。搞定了。这样的话，应该和其他模型的接口是一样的。

