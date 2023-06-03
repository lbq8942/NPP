import torch
import torch.nn as nn
import torch.nn.functional as tnf
from torch.autograd import grad
import time

class MLP(nn.Module):

    def __init__(self,input_dim,hidden_dim,output_dim,dropout=0.05):
        super(MLP,self).__init__()
        self.linear1=nn.Linear(input_dim,hidden_dim)
        self.hidden_drop=nn.Dropout(p=0.05)
        self.linear2=nn.Linear(hidden_dim,output_dim)

    def forward(self,x):
        
        x=self.linear1(x)
        x=tnf.relu(x)
        x=self.hidden_drop(x)
        x=self.linear2(x)
        return x
class TimeParaRNN(nn.Module):

    def __init__(self,input_dim,hidden_dim,output_dim,flownum,flowlen,num_marks):
        super(TimeParaRNN,self).__init__()

        self.flownum,self.flowlen,self.num_marks=flownum,flowlen,num_marks

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.linear1=nn.Linear(input_dim,hidden_dim)
        self.linear2=nn.Linear(hidden_dim,hidden_dim)
        self.linear_weight=nn.Linear(hidden_dim,output_dim)
        self.linear_bias=nn.Linear(hidden_dim,output_dim)
        self.hidden_drop=nn.Dropout(p=0.10)
        



    def forward(self,x):
        
        flownum, flowlen,num_marks=self.flownum,self.flowlen,self.num_marks
        x=self.linear1(x)
        x=tnf.relu(x)
        x=self.linear2(x)
        x=tnf.relu(x)
        x=self.hidden_drop(x)
        weight, bias=self.linear_weight(x),self.linear_bias(x)
        bsize=len(weight)
        
        weight=weight.view(-1,flownum,flowlen)
        
        bias=bias.view(-1,flownum,flowlen)
        weight=weight**2+1e-7
        
        
        
        
        
        
        return weight,bias

class TimePara(nn.Module):

    def __init__(self,input_dim,hidden_dim,fnum,num_marks):
        super(TimePara,self).__init__()

        self.fnum,self.num_marks=fnum,num_marks

        self.linear1=nn.Linear(input_dim,hidden_dim)
        self.linear2=nn.Linear(hidden_dim,hidden_dim)
        self.linear_weight=nn.Linear(hidden_dim,self.fnum*num_marks)
        self.linear_bias=nn.Linear(hidden_dim,self.fnum*num_marks)
        self.hidden_drop=nn.Dropout(p=0.10)
        



    def forward(self,x):
        
        fnum,num_marks=self.fnum,self.num_marks
        x=self.linear1(x)
        x=tnf.relu(x)
        x=self.linear2(x)
        x=tnf.relu(x)
        x=self.hidden_drop(x)
        weight, bias=self.linear_weight(x),self.linear_bias(x)
        bsize=len(weight)
        
        
        
        
        weight=weight**2+1e-5
        bias=bias**2+1e-5
        
        
        
        
        
        
        return weight.view(bsize,-1,num_marks,fnum),bias.view(bsize,-1,num_marks,fnum)


class activation():
    def __init__(self,beta=1,th=20,device=None):
        self.beta=beta
        self.th=th
        self.fact=nn.Softplus(beta=beta,threshold=th)
        thx=torch.tensor([th/beta])
        self.thy=self.fact(thx).to(device)

    def forward(self,x):
        return self.fact(x)
    def backward(self,y):
        mask=y<=self.thy
        x=y.clone()
        x[mask]=torch.log(torch.exp(self.beta*y[mask])-1)/self.beta
        return x

class CDF():
    def __init__(self,a,b):
        self.a=a
        self.b=b

    def forward(self,t):
        
        fnum=self.b.shape[-1]
        len=fnum//2
        
        
        
        
        
        
        
        b1=self.b[...,:len]
        
        b3=self.b[...,len:]

        a1=self.a[...,:len]
        
        a3=self.a[...,len:]

        
        incF1=b1*(torch.exp(a1*t)-1)

        
        
        incF4=a3*torch.pow(t,b3)
        incF=torch.sum(torch.cat([incF1,incF4],dim=-1),dim=-1)
        F=torch.tanh(incF)
        
        return F
    
    
    
    
    
    
    

    def pdf(self,t,training=True):

        t.requires_grad=True
        if training==False:
            torch.set_grad_enabled(True)
            F=self.forward(t)
            f=grad(F.sum(),t)[0]
            torch.set_grad_enabled(False)
        else:
            F=self.forward(t)
            
            f=grad(F.sum(),t,create_graph=True, retain_graph=True)[0]
        return F,f

    def inverse(self,F):
        
        
        flowlen=self.flowlen
        flownum=self.flownum
        
        preF=torch.arctanh(F)
        bsize=len(preF)
        preF=torch.sub(preF.unsqueeze(1).repeat(1,flownum),self.Fsec)
        flowindex=torch.logical_and(preF>=self.tsec[:,0],preF<=self.tsec[:,1])
        flowindex=torch.max(flowindex,dim=-1,keepdim=True)[1]
        
        
        preF=torch.gather(preF,1,flowindex)
        for i in range(flowlen):
            j=flowlen-1-i
            wi=torch.gather(self.weight[...,j],1,flowindex)
            bi=torch.gather(self.bias[...,j],1,flowindex)
            preF=torch.sub(preF,bi)
            preF=torch.div(preF,wi)
            if (i+1)!=flowlen:
                assert torch.isnan(preF).sum()==0
                preF=self.iact(preF)
                assert torch.isnan(preF).sum()==0
            
        
        
        assert torch.isnan(preF).sum()==0
        t=preF*self.std_inter_time+self.mean_inter_time
        return t.view(-1)

    def sample(self):
        
        bsize=len(self.weight)

        test=True
        if test:
            tss=[]
            for F in samples:
                
                cF=torch.tensor([F]).repeat(bsize)
                t=self.inverse(cF)
                tss.append(t)
            return tss

        randomF=torch.rand(bsize,device=self.weight.device)
        
        t=self.inverse(randomF)
        return t

    def predict(self):
        pass