from math import ceil

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical


def vectors_to_demo(uu, vv, ww, device):
    mul_tensor = torch.zeros((4, 4, 4), device=device)
    for i in torch.arange(uu.shape[0]):
        mul_tensor += torch.einsum("p,qr->pqr", uu[i], torch.outer(vv[i], ww[i]))
    # convert to steps/actions
    steps_wide = torch.cat((uu, vv, ww), dim=1)
    steps_wide += 1
    return mul_tensor, steps_wide


def steps_wide_to_uvw(steps_wide, n=4):
    uu, vv, ww = torch.split(steps_wide, n, dim=1)
    return uu, vv, ww


def get_strassen(device: str):
    # uu_short = torch.tensor([[1, 0, 1, 0], [-1, 0, 1, 1]], device=device)
    # vv_short = torch.tensor([[1, -1, 0, 0], [1, 0, 0, 1]], device=device)
    # ww_short = torch.tensor([[-1, 0, 0, 0], [-1, 0, -1, 0]], device=device)
    uu_strassen = torch.tensor(
        [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1],
        ],
        device=device,
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
        ],
        device=device,
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
        ],
        device=device,
    )
    return vectors_to_demo(uu_strassen, vv_strassen, ww_strassen, device)


# strassen_tensor, strassen_steps = vectors_to_demo(uu_strassen, vv_strassen, ww_strassen)
