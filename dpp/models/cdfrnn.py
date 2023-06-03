import torch.nn as nn
import torch
import dpp
from dpp.models.branch import *
from torch.nn.functional import one_hot
from dpp.utils import diff
import torch.nn.functional as tnf

class Attention(nn.Module):
    def __init__(self,args,inputdim,hdim,headsnum=4,num_marks=2):
        super(Attention,self).__init__()
        self.num_marks=num_marks
        self.inputdim=inputdim
        self.hdim=hdim
        self.headsnum=headsnum
        self.headsdim=hdim//headsnum
        self.wq=nn.Linear(args.mdim,hdim)
        self.wk=nn.Linear(args.mdim,hdim)

    def forward(self,q,k):
        
        
        
        
        qx=self.wq(q)
        kx=self.wk(k)
        
        qk=torch.matmul(kx,qx.T)
        
        

        
        
        
        
        
        
        

        
        qk=tnf.normalize(qk, dim=-1)


        return qk



class TPPCDF(nn.Module):
    """
    RNN-based TPP model for marked and unmarked event sequences.

    The marks are assumed to be conditionally independent of the inter-event times.

    Args:
        num_marks: Number of marks (i.e. classes / event types)
        mean_log_inter_time: Average log-inter-event-time, see dpp.data.dataset.get_inter_time_statistics
        std_log_inter_time: Std of log-inter-event-times, see dpp.data.dataset.get_inter_time_statistics
        context_size: Size of the context embedding (history embedding)
        mark_embedding_size: Size of the mark embedding (used as RNN input)
        rnn_type: Which RNN to use, possible choices {"RNN", "GRU", "LSTM"}

    """
    def __init__(
        self,
        num_marks: int,
        mean_log_inter_time: float = 0.0,
        std_log_inter_time: float = 1.0,
        context_size: int = 32,
        mark_embedding_size: int = 32,
        rnn_type: str = "GRU",
        flownum:int =30,
        flowlen:int=2,
        
            args=None
    ):
        super().__init__()
        self.num_marks = num_marks
        self.args=args
        self.mean_log_inter_time = mean_log_inter_time
        self.std_log_inter_time = std_log_inter_time
        self.context_size = context_size
        self.mark_embedding_size = mark_embedding_size
        self.flownum=flownum
        self.flowlen=flowlen
        
        if self.num_marks > 1:
            self.num_features = 1 + self.mark_embedding_size
            self.mark_embedding = nn.Embedding(self.num_marks, self.mark_embedding_size)
            self.markp = MLP(self.context_size,self.context_size, self.num_marks)  
            
        else:
            self.num_features = 1
        self.rnn_type = rnn_type
        self.context_init = nn.Parameter(torch.zeros(context_size))  
        self.rnn = getattr(nn, rnn_type)(input_size=self.num_features, hidden_size=self.context_size, batch_first=True)
        self.timepara=TimePara(self.context_size,self.context_size,self.flownum,self.num_marks)
        


    def get_features(self, batch):
        """
        Convert each event in a sequence into a feature vector.

        Args:
            batch: Batch of sequences in padded format (see dpp.data.batch).

        Returns:
            features: Feature vector corresponding to each event,
                shape (batch_size, seq_len, num_features)

        """
        features = batch.inter_times.unsqueeze(2)
        features = (features - self.mean_log_inter_time) / self.std_log_inter_time
        mark_emb=self.mark_embedding(batch.marks)
        features=torch.cat([features,mark_emb],dim=-1)
        
        
        return features  

    def get_context(self, features,remove_last: bool = True) -> torch.Tensor:
        """
        Get the context (history) embedding from the sequence of events.

        Args:
            features: Feature vector corresponding to each event,
                shape (batch_size, seq_len, num_features)
            remove_last: Whether to remove the context embedding for the last event.

        Returns:
            context: Context vector used to condition the distribution of each event,
                shape (batch_size, seq_len, context_size) if remove_last == False
                shape (batch_size, seq_len + 1, context_size) if remove_last == True

        """
        
        

        context = self.rnn(features)[0]
        batch_size, seq_len, context_size = context.shape
        context_init = self.context_init[None, None, :].expand(batch_size, 1, -1)  
        
        if remove_last:
            context = context[:, :-1, :]
        context = torch.cat([context_init, context], dim=1)

        return context

    def log_prob(self,batch,test=False):  
        """Compute log-likelihood for a batch of sequences.

        Args:
            batch:

        Returns:
            log_p: shape (batch_size,)

        """
        flownum=self.flownum
        flowlen=self.flowlen
        num_marks=self.num_marks
        features = self.get_features(batch)  
        context = self.get_context(features)  

        a,b = self.timepara(context)  
        
        bsize = context.shape[0]  
        cdf = CDF(a, b,self.args)  
        inter_times = batch.inter_times  
        
        inter_times_marks = inter_times.unsqueeze(-1).repeat(1, 1, num_marks).unsqueeze(-1)  
        inter_times_marks = torch.clamp(inter_times_marks,1e-5)  

        F, f = cdf.pdf(inter_times_marks, training=self.training)  
        
        log_p = torch.log(f + 1e-8).squeeze(-1)
        multi_marks = one_hot(batch.marks,num_classes=self.num_marks).float()  
        pos_log_p = log_p * multi_marks  
        

        if test:  
            
            
            times = torch.zeros_like(inter_times_marks)
            
            bsize, seqlen, m, pad = inter_times_marks.shape
            
            times_max = 10 * torch.ones_like(inter_times_marks)  
            times_min = torch.zeros_like(inter_times_marks)
            times = times_max.clone()

            Fi, fi = cdf.pdf(times, training=self.training)  
            assert torch.any(Fi > 0.5)
            flag = True
            while flag:
                Fi, fi = cdf.pdf(times, training=self.training)  
                Fi = Fi.unsqueeze(-1)
                times = times.detach()
                
                if torch.all((Fi - 0.5) < 0.005):
                    flag = False
                times_max = torch.where(Fi < 0.5, times_max, times)
                times_min = torch.where(Fi > 0.5, times_min, times)
                times = (times_max + times_min) / 2
        
        mark_logits = torch.log_softmax(self.markp(context),dim=-1)  
        tot_nll = (pos_log_p + mark_logits * multi_marks).sum(-1) * batch.masks  
        tot_nll = tot_nll.sum(-1)   
        
        mark_pred=(( mark_logits).argmax(
            -1) * batch.masks).float()
        mark_class_joint = ((log_p + mark_logits).argmax(
            -1) * batch.masks).float()  
        
        time_pred=0
        if test:
            F=(F*multi_marks).sum(dim=-1)* batch.masks
            pred_times=((times.squeeze(-1))*multi_marks).sum(dim=-1)* batch.masks
        else:
            F=None
            pred_times=None
        return tot_nll, mark_pred,time_pred,F,pred_times


    def sample(self, t_end: float, batch_size: int = 1, context_init: torch.Tensor = None):
        """Generate a batch of sequence from the model.

        Args:
            t_end: Size of the interval on which to simulate the TPP.
            batch_size: Number of independent sequences to simulate.
            context_init: Context vector for the first event.
                Can be used to condition the generator on past events,
                shape (context_size,)

        Returns;
            batch: Batch of sampled sequences. See dpp.data.batch.Batch.
        """
        device=self.timepara.linear_weight.weight.device
        if context_init is None:
            
            context_init = self.context_init
        else:
            
            context_init = context_init.view(self.context_size)
        next_context = context_init[None, None, :].expand(batch_size, 1, -1)
        inter_times = torch.empty(batch_size, 0,device=device)
        marks = torch.empty(batch_size, 0, dtype=torch.long,device=device)

        generated = False
        flownum,flowlen,num_marks=self.flownum,self.flowlen,self.num_marks
        while not generated:
            weight, bias = self.timepara(next_context)  
            
            bsize = next_context.shape[0]  
            cdf = CDF(weight, bias, flownum, flowlen, num_marks,self.mean_inter_time,self.std_inter_time)  
            t=cdf.sample()
            next_inter_times=t.reshape(-1,1,num_marks)
            
            
            mark_logits = torch.log_softmax(self.markp(next_context),dim=-1)  
            mark_dist = Categorical(logits=mark_logits)
            next_marks = mark_dist.sample()  
            marks = torch.cat([marks, next_marks], dim=1)
            
            next_inter_time = torch.gather(next_inter_times.squeeze(), dim=-1, index=next_marks) 
            inter_times = torch.cat([inter_times, next_inter_time], dim=-1)  

            with torch.no_grad():
                generated = inter_times.sum(-1).min() >= t_end

            batch = Batch(inter_times=inter_times, mask=torch.ones_like(inter_times), marks=marks)
            features = self.get_features(batch)  
            context = self.get_context(features, remove_last=False)  
            next_context = context[:, [-1], :]  
        
        arrival_times = inter_times.cumsum(-1)  
        inter_times = diff(arrival_times.clamp(max=t_end), dim=-1)
        mask = (arrival_times <= t_end).float()  
        if self.num_marks > 1:
            marks = marks * mask  
        return Batch(inter_times=inter_times, mask=mask, marks=marks)
