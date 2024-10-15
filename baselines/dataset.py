
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.utils.data
from utils import pad_sequence

def load_data(args):
    dataset_name = args.data
    batch_size = args.bs

    dataset,t_end = load_dataset(dataset_name,args)
    args.t_end,args.num_seq=t_end,len(dataset)
    d_train, d_val, d_test = dataset.train_val_test_split()

    dl_train = d_train.get_dataloader(batch_size=batch_size, shuffle=True)
    dl_val = d_val.get_dataloader(batch_size=batch_size, shuffle=False)
    dl_test = d_test.get_dataloader(batch_size=batch_size, shuffle=False)
    
    args.m=d_train.num_marks
    
    args.mean_inter_time, args.std_inter_time = d_train.get_inter_time_statistics()  
    args.mintime = max(0, args.mean_inter_time - args.std_inter_time * args.cover)
    args.maxtime = args.mean_inter_time + args.std_inter_time * args.cover  

    return dataset, d_train, d_val, d_test, dl_train, dl_val, dl_test


class Sequence:
    def __init__(self,inter_times,marks):
        self.inter_times=torch.tensor(inter_times,dtype=torch.float32)
        self.marks=torch.tensor(marks,dtype=torch.int64)
        

class Batch():
    def __init__(self,inter_times,marks,masks):
        self.inter_times=inter_times
        self.marks=marks
        self.masks=masks



def from_list(sequences: List[Sequence]):
    batch_size = len(sequences)
    
    
    max_seq_len = max(len(seq.inter_times) for seq in sequences)
    inter_times = pad_sequence([seq.inter_times for seq in sequences], max_len=max_seq_len)

    dtype = sequences[0].inter_times.dtype
    device = sequences[0].inter_times.device
    mask = torch.zeros(batch_size, max_seq_len, device=device, dtype=dtype)

    for i, seq in enumerate(sequences):
        mask[i, :len(seq.inter_times)] = 1

    if sequences[0].marks is not None:
        marks = pad_sequence([seq.marks for seq in sequences], max_len=max_seq_len)
    else:
        marks = None

    

    return Batch(inter_times, marks,mask)

def time_statistics(dataset,get_inter_times):
    seqtimes=[np.log(get_inter_times(seq)+1) for seq in dataset["sequences"]]
    xx=np.concatenate(seqtimes)
    print(xx.mean())
    print(xx.std())
    print(xx.max())

def load_dataset(name: str,args):
    if not name.endswith(".pkl"):
        name += ".pkl"
    path_to_file = args.data_path+"/{}/{}".format(args.datadir,name)
    dataset = torch.load(str(path_to_file))

    
    if args.proportion<=1:
        wantnum=int(len(dataset["sequences"])*args.proportion)
    else:
        wantnum=int(args.proportion)
    if wantnum<5:
        args.logger.info("not enough sequences to train,numseqs<5")
    
    dataset["sequences"]=dataset["sequences"][:wantnum]

    def get_inter_times(seq: dict):
        """Get inter-event times from a sequence."""
        return np.ediff1d(np.concatenate([[0], seq["arrival_times"]]))

    
    sequences = [
        Sequence(
            inter_times=np.log(get_inter_times(seq)+1),
            marks=seq.get("marks"),
        )
        for seq in dataset["sequences"]
    ]
    t_end=dataset["sequences"][0]["t_end"]
    return SequenceDataset(sequences=sequences, num_marks=dataset.get("num_marks", 1)),t_end


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, sequences: List[Sequence], num_marks=1):
        self.sequences = sequences
        self.num_marks = num_marks

    def __getitem__(self, item):
        return self.sequences[item]

    def __len__(self):
        return len(self.sequences)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self)})"

    def __add__(self, other: "SequenceDataset") -> "SequenceDataset":
        if not isinstance(other, SequenceDataset):
            raise ValueError(f"other must be a SequenceDataset (got {type(other)})")
        new_num_marks = max(self.num_marks, other.num_marks)
        new_sequences = self.sequences + other.sequences
        return SequenceDataset(new_sequences, num_marks=new_num_marks)

    def get_dataloader(
            self, batch_size: int = 32, shuffle: bool = True
    ) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, collate_fn=from_list
        )

    def train_val_test_split(
            self, train_size=0.6, val_size=0.2, test_size=0.2, seed=None, shuffle=True,
    ) -> Tuple["SequenceDataset", "SequenceDataset", "SequenceDataset"]:
        """Split the sequences into train, validation and test subsets."""
        if train_size < 0 or val_size < 0 or test_size < 0:
            raise ValueError("train_size, val_size and test_size must be >= 0.")
        if train_size + val_size + test_size != 1.0:
            raise ValueError("train_size, val_size and test_size must add up to 1.")

        if seed is not None:
            np.random.seed(seed)

        all_idx = np.arange(len(self))
        if shuffle:
            np.random.shuffle(all_idx)

        train_end = int(train_size * len(self))  
        val_end = int((train_size + val_size) * len(self))  

        train_idx = all_idx[:train_end]
        val_idx = all_idx[train_end:val_end]
        test_idx = all_idx[val_end:]

        train_sequences = [self.sequences[idx] for idx in train_idx]
        val_sequences = [self.sequences[idx] for idx in val_idx]
        test_sequences = [self.sequences[idx] for idx in test_idx]

        return (
            SequenceDataset(train_sequences, num_marks=self.num_marks),
            SequenceDataset(val_sequences, num_marks=self.num_marks),
            SequenceDataset(test_sequences, num_marks=self.num_marks),
        )

    def get_inter_time_statistics(self):
        """Get the mean and std of inter_time."""
        all_inter_times = torch.cat([seq.inter_times for seq in self.sequences])
        mean_inter_time = all_inter_times.mean()
        std_inter_time = all_inter_times.std()
        return mean_inter_time, std_inter_time

    @property
    def total_num_events(self):
        return sum(len(seq.inter_times) for seq in self.sequences)



