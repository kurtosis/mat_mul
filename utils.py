from typing import List, Tuple

import torch


def print_params(model):
    print(f"{sum(p.numel() for p in model.parameters()) // int(1e6)}M parameters")
    print(f"{sum(p.numel() for p in model.parameters()) // int(1e3)}k parameters")
    print(f"{sum(p.numel() for p in model.torso.parameters())} parameters: torso")
    print(
        f"{sum(p.numel() for p in model.policy_head.parameters()) // int(1e6)}M parameters: policy head"
    )
    print(
        f"{sum(p.numel() for p in model.policy_head.parameters())} parameters: policy head"
    )
    print(
        f"{sum(p.numel() for p in model.value_head.parameters())} parameters: value head"
    )


def get_scalars(tt: torch.Tensor, t_step: int, batch_size=True):
    """
    Args:
        tt: Tensor of shape (batch_size, dim_t, dim_3d, dim_3d, dim_3d)
        t_step: Time step
        batch_size: Whether the input tensor is batched or single example
    Returns:
        scalars: Tensor of shape (batch_size, 1)
    """
    if batch_size:
        b = tt.shape[0]
        scalars = torch.zeros((b, 1))
        scalars[:, 0] = t_step
    else:
        scalars = torch.tensor(t_step).unsqueeze(0).float()
    return scalars


def uvw_to_demo(
    uu: torch.Tensor, vv: torch.Tensor, ww: torch.Tensor, device: str, shift=1
):
    """Converts a triplet of factor lists to the tensor and action_list.
    Assumes factor values are in {-1, 0, 1} and shifts factor tokens to start at 0."""
    mult_tensor = torch.zeros((4, 4, 4), device=device)
    for i in torch.arange(uu.shape[0]):
        mult_tensor += (
            uu[i].view(-1, 1, 1) * vv[i].view(1, -1, 1) * ww[i].view(1, 1, -1)
        )
    # convert to steps/actions
    action_list = torch.cat((uu, vv, ww), dim=1)
    action_list += shift
    return mult_tensor, action_list


def action_to_uvw(action: torch.Tensor, shift=1):
    """Convert the standard representation of an action (or batch of actions) to the factor representation.
    Args:
        action: Tensor of shape (*, 3*dim_3d) (action may be a singleton or batch)
        shift: Shift factor values by this amount
    Returns:
        uvw: Tuple of 3 Tensors of shape (*, dim_3d)
    """
    # assert len(action.shape) <= 2
    dim_3d = action.shape[-1] // 3
    uvw = (action - shift).split(dim_3d, dim=-1)
    # uvw = [x.squeeze() for x in uvw]
    return uvw


def uvw_to_tensor(uvw: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
    """Convert the factor representation of an action to the tensor representation.
    Args:
        uvw: Tuple of 3 Tensors of shape (dim_3d)
    Returns:
        tensor_action: Tensor of shape (dim_3d, dim_3d, dim_3d)
    """
    uu, vv, ww = uvw
    if uu.dim() == 1:
        tensor_action = uu.view(-1, 1, 1) * vv.view(1, -1, 1) * ww.view(1, 1, -1)
    else:
        tensor_action = (
            uu.unsqueeze(-1).unsqueeze(-1)
            * vv.unsqueeze(-1).unsqueeze(-3)
            * ww.unsqueeze(-2).unsqueeze(-3)
        )
    return tensor_action


def action_to_tensor(action: torch.Tensor):
    """Convert the standard representation of an action to the tensor representation.
    Args:
        action: Tensor of shape (3*dim_3d)
    Returns:
        tensor_action: Tensor of shape (dim_3d, dim_3d, dim_3d)
    """
    uvw = action_to_uvw(action)
    return uvw_to_tensor(uvw)


def get_head_state(state: torch.Tensor, unsqueeze=True):
    """
    Args:
        state: Tensor of shape (batch_size, dim_t, dim_3d, dim_3d, dim_3d)
    Returns:
        head_state: Tensor of shape (batch_size, 1, dim_3d, dim_3d, dim_3d)
                                    or (batch_size, dim_3d, dim_3d, dim_3d)
                                    the current state of matmul tensor
    """
    if unsqueeze:
        return state[:, 0].unsqueeze(1)
    else:
        return state[:, 0]


def update_state(state: torch.Tensor, action: torch.Tensor, batch=True):
    """Update the state by applying the action.
    Args:
        state: Tensor of shape (batch_size, dim_t, dim_3d, dim_3d, dim_3d)
        action: Tensor of shape (batch_size, n_samples, 3*dim_3d)
    Returns:
        new_state: Tensor of shape (batch_size, dim_t, dim_3d, dim_3d, dim_3d)
    """
    if batch:
        tensor_action = action_to_tensor(action.squeeze())
        new_tensor = state[:, 0] + tensor_action
        new_tensor = new_tensor.unsqueeze(1)
        new_state = torch.cat((new_tensor, state[:, :-1]), dim=1)
    else:
        tensor_action = action_to_tensor(action)
        new_tensor = get_head_state(state) + tensor_action
        new_state = torch.cat((new_tensor, state[:-1]), dim=0)
    return new_state


def get_rank(state: torch.Tensor):
    headstate = get_head_state(state, unsqueeze=False)
    # return int(torch.linalg.matrix_rank(current_state).sum())
    return torch.linalg.matrix_rank(headstate).sum()


def build_matmul_tensor(dim_t: int, dim_i: int, dim_j: int, dim_k: int):
    """Build a tensor representing the matrix multiplication AB=C, where:
    A ~ (dim_i, dim_j)
    B ~ (dim_j, dim_k)
    C ~ (dim_i, dim_k)
    Args:
        dim_t (int): The tensor memory length (including the current state).
        dim_i (int): The first dimension of matrix A and of matrix C.
        dim_j (int): The second dimension of matrix A and the first dimension of matrix B.
        dim_k (int): The second dimension of matrix B and of matrix C.
    Returns:
        A tensor of shape (dim_t, dim_i*dim_j, dim_j*dim_k, dim_i*dim_k) representing the multiplication AB=C.
        The tensor has a value of 1 at every index (0, l, m, n) where a_l*b_m --> c_n (for flattened indexes).
    """
    matmul_tensor = torch.zeros(dim_t, dim_i * dim_j, dim_j * dim_k, dim_i * dim_k)
    for ik in range(dim_i * dim_k):
        for j in range(dim_j):
            matmul_tensor[0, (ik // dim_j) * dim_k + j, j * dim_j + ik % dim_j, ik] = 1
    return matmul_tensor


def state_to_str(state: torch.Tensor):
    """Converts a state tensor to a string, suitable for dict key."""
    string = "_".join(
        state.reshape(-1).long().detach().cpu().numpy().astype(str).tolist()
    )
    return string


def str_to_state(string: str, shape: tuple) -> torch.Tensor:
    """Converts a string back to the original tensor.
    Args:
        string (str): String version of tensor.
        shape (tuple): The shape of the original tensor.
    """
    return torch.tensor([float(x) for x in string.split("_")]).reshape(shape)


def tensor_factorized(state):
    """Determines whether tensor has been factorized.
    TO DO: figure out all the dimensions we need.
    Args:
        state (torch.Tensor): The state of the game.
    """
    # state size (*, S, S, S)
    return (state[0] == 0).all()


def remove_null_actions(state: torch.Tensor, candidate_states: List[torch.Tensor]):
    """From a list of candidate states, return indexes of entries that differ from original state."""
    idxs = [i for i, c in enumerate(candidate_states) if (c[:, 0] != state[:, 0]).any()]
    return idxs
