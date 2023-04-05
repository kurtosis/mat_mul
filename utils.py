import torch


def get_scalars(tt: torch.Tensor, t_step: int, batch_size = True):
    if batch_size:
        b = tt.shape[0]
        scalars = torch.zeros((b, 1))
        scalars[:, 0] = t_step
    else:
        scalars = torch.tensor(t_step).unsqueeze(-1).float()
    return scalars


def print_params(alpha):
    print(f"{sum(p.numel() for p in alpha.parameters()) // int(1e6)}M parameters")
    print(f"{sum(p.numel() for p in alpha.parameters()) // int(1e3)}k parameters")
    print(f"{sum(p.numel() for p in alpha.torso.parameters())} parameters: torso")
    print(
        f"{sum(p.numel() for p in alpha.policy_head.parameters()) // int(1e6)}M parameters: policy head"
    )
    print(
        f"{sum(p.numel() for p in alpha.policy_head.parameters())} parameters: policy head"
    )
    print(
        f"{sum(p.numel() for p in alpha.value_head.parameters())} parameters: value head"
    )
