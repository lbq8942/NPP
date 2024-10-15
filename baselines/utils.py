import torch
from typing import Any, List, Optional
def pad_sequence(
        sequences: List[torch.Tensor],
        padding_value: float = 0,
        max_len: Optional[int] = None,
):
    r"""Pad a list of variable length Tensors with ``padding_value``"""
    dtype = sequences[0].dtype
    device = sequences[0].device
    seq_shape = sequences[0].shape
    trailing_dims = seq_shape[1:]
    if max_len is None:
        max_len = max([s.size(0) for s in sequences])
    out_dims = (len(sequences), max_len) + trailing_dims

    out_tensor = torch.empty(*out_dims, dtype=dtype, device=device).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        
        out_tensor[i, :length, ...] = tensor

    return out_tensor

def updata_args(args,parameters):
    
    para=args.baseargs
    arg_dict = args.__dict__
    args.__dict__.update(parameters)
    paradict = {}
    if para != None:
        keyvalues=para.split(",")
        for keyvalue in keyvalues:
            key,value=keyvalue.split(":")
            arg_dict[key]=type(arg_dict[key])(value)
    
    
    return args