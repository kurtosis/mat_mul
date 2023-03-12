from math import ceil

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical


uu_short = torch.tensor([[1, 0, 1, 0], [-1, 0, 1, 1]])
vv_short = torch.tensor([[1, -1, 0, 0], [1, 0, 0, 1]])
ww_short = torch.tensor([[-1, 0, 0, 0], [-1, 0, -1, 0]])

uu_strassen = torch.tensor(
    [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
)

vv_strassen = torch.tensor(
    [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
)

ww_strassen = torch.tensor(
    [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]
)


def vectors_to_demo(uu, vv, ww):
    mul_tensor = torch.zeros((4, 4, 4))
    for i in torch.arange(uu.shape[0]):
        mul_tensor += torch.einsum("p,qr->pqr", uu[i], torch.outer(vv[i], ww[i]))
    # convert to steps/actions
    steps_wide = torch.cat((uu, vv, ww), dim=1)
    steps_wide += 1
    return mul_tensor, steps_wide


def steps_wide_to_uvw(steps_wide, n=4):
    uu, vv, ww = torch.split(steps_wide, n, dim=1)
    return uu, vv, ww


strassen_tensor, strassen_steps = vectors_to_demo(uu_strassen, vv_strassen, ww_strassen)
