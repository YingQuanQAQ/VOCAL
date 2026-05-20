from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import numpy as np


def _numpy_dtype_to_torch(dtype):
    return torch.from_numpy(np.empty((), dtype=dtype)).dtype


def to_torch(x, use_gpu=True, dtype=np.float32):
    torch_dtype = _numpy_dtype_to_torch(dtype)

    if torch.is_tensor(x):
        var = x.to(dtype=torch_dtype)
    elif isinstance(x, (list, tuple)) and x and all(torch.is_tensor(v) for v in x):
        var = torch.stack([v.to(dtype=torch_dtype) for v in x])
    else:
        x = np.array(x, dtype=dtype)
        var = torch.from_numpy(x)

    return var.cuda() if use_gpu else var.cpu()


def to_numpy(x):
    if isinstance(x, int) or isinstance(x, float):
        return x
    if isinstance(x, (list, np.ndarray)):
        return np.array([to_numpy(_x) for _x in x])
    return x.detach().cpu().numpy()


def norm_col_init(weights, std=1.0):
    """
    Normalized column initializer
    """
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x ** 2).sum(1, keepdim=True))
    return x
