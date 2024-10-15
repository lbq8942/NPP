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
        # use index notation to prevent duplicate references to the tensor
        out_tensor[i, :length, ...] = tensor

    return out_tensor

def updata_args(args,parameters):
    #这里需要将args给说清楚。args是一个字典格式--args d:daf,的形式。
    para=args.baseargs
    arg_dict = args.__dict__
    args.__dict__.update(parameters)#注意一下更新顺序，先填入默认的，然后更新我们填入的。
    paradict = {}
    if para != None:
        keyvalues=para.split(",")#可以得到这么多个参数。不对啊，其实我们还应该和那个子args合并，否则是不行的，所以一并放到
        for keyvalue in keyvalues:
            key,value=keyvalue.split(":")
            arg_dict[key]=type(arg_dict[key])(value)#数据转换，这里完成更新。
    # arg_dict.update(paradict)  #更改这个字典就等价于更改args，十分方便。的确更改了，但是一个问题，数据结构的类型改变了，比如整数，浮点数之类的。
    #这里只是暂时更新而已，后面还要继续更新。似乎这里不应该更新，而应该到后面的时候再更新。
    return args