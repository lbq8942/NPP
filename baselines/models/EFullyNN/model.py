
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as tnf
from torch.autograd import grad

model_parameters={
    "addtimelayer":1,
    "coeftimeloss": 0.01,  
    "residual":0
}

def updata_args(args):
    
    para=args.baseargs
    arg_dict = args.__dict__
    args.__dict__.update(model_parameters)
    paradict = {}
    if para != None:
        keyvalues=para.split(",")
        for keyvalue in keyvalues:
            key,value=keyvalue.split(":")
            arg_dict[key]=type(arg_dict[key])(value)
    
    
    return args


class Encoder(nn.Module):
    def __init__(self,args):
        super(Encoder,self).__init__()
        self.args=args
        self.arch = nn.GRU(input_size=args.mdim+1, hidden_size=args.hdim,batch_first=True)
        
    def forward(self,x):
        
        context = self.arch(x)[0][:, :-1, :]
        
        
        
        
        
        return context

class MLP(nn.Module):
    def __init__(self,dims):
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
            if (i+1)!=lnum:
                x=act(x)
        return x

def x3(x):
    return torch.pow(x,3)

class Decoder(nn.Module):
    def __init__(self,args):
        super(Decoder,self).__init__()
        
        
        self.hazard_nn_h1= nn.Linear(args.hdim,args.hdim)
        if args.residual:
            self.residual_nn=nn.Linear(args.hdim, 1)

        self.hazard_nn_t1 = nn.Parameter(torch.rand(args.hdim))

        self.hazard_nn=MLP([args.hdim, args.hdim, 1])
        
        self.type_nn=MLP([args.hdim+1,args.hdim,args.m])
        if args.addtimelayer:
            self.time_nn=MLP([args.hdim,args.hdim,1])
        self.args = args
        self.actfn=torch.tanh
        

        
        
        

    def make_pos(self):
        self.hazard_nn_t1.data*= (self.hazard_nn_t1.data>=0)
        for p in self.hazard_nn.parameters():
            p.data *= (p.data >= 0)  

    def get_hazard(self,h,t):
        if len(t.shape)==3:
            t=t.unsqueeze(3)
        else:
            t = t.unsqueeze(2).unsqueeze(3)

        h2 = self.hazard_nn_h1(h)  
        t2 = (self.hazard_nn_t1.unsqueeze(0).unsqueeze(0).unsqueeze(0)) *t
            

        ht2 = self.actfn(h2.unsqueeze(2) + t2)  
        hazardt = self.hazard_nn(ht2,
                                 act=self.actfn)  
        
        if self.args.residual:
            res = self.residual_nn(h)
            residual = torch.relu(res[..., [-1]].unsqueeze(2)) * t  
            hazardt=residual+hazardt
        return hazardt

    def forward(self,h,t):
        
        self.make_pos()
        
        t.requires_grad_(True)

        if self.training==False:
            torch.set_grad_enabled(True)
            hazardt = self.get_hazard(h, t)
            hazard0 = self.get_hazard(h, torch.zeros_like(t))
            hazard = (hazardt - hazard0).squeeze(3)  
            intensity=grad(hazard.sum(),t)[0]
            torch.set_grad_enabled(False)
        else:
            hazardt = self.get_hazard(h, t)
            hazard0 = self.get_hazard(h, torch.zeros_like(t))
            hazard = (hazardt - hazard0).squeeze(3)  
            intensity = grad(hazard.sum(), t, create_graph=True, retain_graph=True)[0]
        
        
        intensity=intensity+self.args.minpositive
        return intensity,hazard

    def ll(self,h,t,marks):
        
        timet=t*self.args.scale
        intensity,hazard=self.forward(h,timet)
        hazard=hazard.squeeze(2)
        marks = tnf.one_hot(marks, num_classes=self.args.m)  
        typet=t*self.args.scale
        typelogits=torch.log_softmax(self.type_nn(torch.cat([h,typet.unsqueeze(2)],dim=-1)),dim=-1)
        
        m_typelogits = (typelogits * marks).sum(dim=-1)  
        tppll=torch.log(intensity)-hazard+m_typelogits+np.log(self.args.scale)
        ll=tppll
        if self.args.addtimelayer:
            time_pred=self.time_nn(h).squeeze(2)
            time_loss=-torch.pow((time_pred - typet), 2)
            ll+=self.args.coeftimeloss*time_loss
        return ll,tppll

    def tpp_prediction(self,h):
        
        num=self.args.monum
        mintime,maxtime=self.args.mintime*self.args.scale,self.args.maxtime*self.args.scale
        interval=(maxtime-mintime)
        
        bs,seqlen,hdim=h.shape
        r=torch.rand(bs,seqlen,num,device=self.args.device)
        samplet=r*interval+mintime
        
        intensity,hazard=self.forward(h,samplet)
        ft=intensity*torch.exp(-hazard)
        
        time_pred=interval*((samplet*ft).mean(dim=-1))
        return time_pred

    
    def prediction(self,h):
        if self.args.addtimelayer:
            time_pred=self.time_nn(h).squeeze(2)
        else:
            time_pred=self.tpp_prediction(h)
        type_pred=torch.argmax(self.type_nn(torch.cat([h,time_pred.unsqueeze(2)],dim=-1)), dim=-1)
        return time_pred/self.args.scale,type_pred

class efullynn(nn.Module):
    def __init__(
        self,
            args
    ):
        super().__init__()
        args=updata_args(args)
        
        if args.maxtime>5:
            args.scale=5/args.maxtime
            
        args.logger.info(args)
        self.num_marks = args.m
        self.args=args
        self.mean_log_inter_time = args.mean_inter_time
        self.std_log_inter_time = args.std_inter_time
        self.mark_embedding_size = args.mdim
        self.mark_embedding=nn.Embedding(args.m,args.mdim)
        self.encoder=Encoder(args)
        self.decoder=Decoder(args)

    def embed(self, batch):

        features = batch.inter_times.unsqueeze(2)
        features = (features - self.mean_log_inter_time) / self.std_log_inter_time
        if self.num_marks > 1:
            mark_emb = self.mark_embedding(batch.marks)  
            features = torch.cat([features, mark_emb], dim=-1)
        return features  

    def log_prob(self,batch,test=False):  

        num_marks=self.num_marks
        features = self.embed(batch)  
        context = self.encoder(features)  
        ll,tppll=self.decoder.ll(context,batch.inter_times[:,1:],batch.marks[:,1:])
        ll = (ll * (batch.masks[:,1:])).sum(dim=-1)  
        
        if test:
            time_pred,type_pred=self.decoder.prediction(context)
            
            
            tppll = (tppll * (batch.masks[:,1:])).sum(dim=-1)
            ll=tppll
        else:
            time_pred=0
            type_pred=0
        
        return ll,time_pred,type_pred

