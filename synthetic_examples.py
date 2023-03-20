import torch


def factors_to_demo(uu: torch.Tensor, vv: torch.Tensor, ww: torch.Tensor, device: str):
    """Converts a triplet of factor lists to the tensor and action_list.
    Assumes factor values are in {-1, 0, 1} and shifts factor tokens to start at 1,
    with 0 reserved for start of sequence token."""
    mult_tensor = torch.zeros((4, 4, 4), device=device)
    for i in torch.arange(uu.shape[0]):
        # mul_tensor += torch.einsum("p,qr->pqr", uu[i], torch.outer(vv[i], ww[i]))
        mult_tensor += (
            uu[i].view(-1, 1, 1) * vv[i].view(1, -1, 1) * ww[i].view(1, 1, -1)
        )
    # convert to steps/actions
    action_list = torch.cat((uu, vv, ww), dim=1)
    action_list += 2
    return mult_tensor, action_list


def action_list_to_factors(action_list, n=4):
    uu, vv, ww = torch.split(action_list - 2, n, dim=-1)
    return uu, vv, ww


def get_strassen_factors(device: str):
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
    return uu_strassen, vv_strassen, ww_strassen


def get_strassen_tensor(device: str):
    uu_strassen, vv_strassen, ww_strassen = get_strassen_factors(device)
    return factors_to_demo(uu_strassen, vv_strassen, ww_strassen, device)


if __name__ == "__main__":
    device = "cpu"
    # uu_strassen, vv_strassen, ww_strassen = get_strassen_factors(device)
    # strassen_tensor, action_list = get_strassen_tensor(device)
    pass
