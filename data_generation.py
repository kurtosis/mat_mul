from pathlib import Path
from typing import List, Tuple

import torch
from torch.distributions.categorical import Categorical
from torch.utils.data import Dataset


SAVE_DIR_SYNTH_DEMOS = Path("data_unversioned/synthetic_demos")


class SyntheticDemoDataset(Dataset):
    """Create a set of synthetic demonstrations and save to disk."""

    def __init__(
        self,
        max_actions: int,
        n_demos: int,
        dim_t: int,
        dim_3d: int,
        device: str,
        values=(-1, 0, 1),
        probs=(0.15, 0.7, 0.15),
        overwrite=True,
        save_dir=SAVE_DIR_SYNTH_DEMOS,
    ):
        super().__init__()
        self.max_actions = max_actions
        self.n_demos = n_demos
        self.dim_t = dim_t
        self.dim_3d = dim_3d
        self.values = torch.tensor(values)
        self.probs = torch.tensor(probs)
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        if overwrite:
            for f in self.save_dir.glob("target_tensor_*.pt"):
                f.unlink()
            for f in self.save_dir.glob("action_list_*.pt"):
                f.unlink()
            n_demos_stored = 0
        else:
            n_demos_stored = len(list(self.save_dir.glob("target_tensor_*.pt")))
        # create demonstrations and save to disk
        if n_demos_stored < n_demos:
            self.len_data = n_demos_stored
            for i_demo, (action_list, target_tensor) in enumerate(
                self._create_synthetic_demos(n_demos - n_demos_stored)
            ):
                torch.save(
                    action_list,
                    self.save_dir.joinpath(f"action_list_{self.len_data}.pt"),
                )
                torch.save(
                    target_tensor,
                    self.save_dir.joinpath(f"target_tensor_{self.len_data}.pt"),
                )
                self.len_data += 1
        else:
            self.len_data = n_demos

    def __len__(self):
        return self.len_data * self.max_actions

    @torch.no_grad()
    def __getitem__(self, idx: int):
        idx_demo = idx // self.max_actions
        idx_action = idx % self.max_actions
        action_list = torch.load(self.save_dir.joinpath(f"action_list_{idx_demo}.pt"))
        target_tensor = torch.load(
            self.save_dir.joinpath(f"target_tensor_{idx_demo}.pt"),
        )
        if idx_action != self.max_actions - 1:
            actions = action_list[idx_action + 1 :]
            target_tensor = self._take_actions(actions, target_tensor)
        action = action_list[idx_action]
        target_tensor = torch.stack(
            [
                target_tensor,
                *(
                    self._action_to_tensor(action)
                    for action in reversed(
                        action_list[idx_action + 1 : idx_action + self.dim_t]
                    )
                ),
            ]
        )
        if len(target_tensor) < self.dim_t:
            target_tensor = torch.cat(
                [
                    target_tensor,
                    torch.zeros(
                        self.dim_t - len(target_tensor),
                        *target_tensor.shape[1:],
                    ),
                ]
            )
        scalar = torch.tensor(self.max_actions - idx_action).unsqueeze(-1).float()
        # policy = torch.cat(action)
        policy = action
        reward = torch.tensor([-(idx_action + 1)], dtype=torch.float32)
        return (
            target_tensor.to(self.device),
            scalar.to(self.device),
            policy.to(self.device),
            reward.to(self.device),
        )

    def _create_synthetic_demos(self, n_demos_needed: int):
        """Generate a mult tensor and a list of actions producing it"""
        for _ in range(n_demos_needed):
            target_tensor = torch.zeros(self.dim_3d, self.dim_3d, self.dim_3d)
            action_list = []
            for i in range(self.max_actions):
                valid_action = False
                while not valid_action:
                    uu = self._factor_sample()
                    vv = self._factor_sample()
                    ww = self._factor_sample()

                    tensor_update = (
                        uu.view(-1, 1, 1) * vv.view(1, -1, 1) * ww.view(1, 1, -1)
                    )
                    if not (tensor_update == 0).all():
                        valid_action = True
                        action_list.append(torch.cat((uu, vv, ww)) + 2)
                        target_tensor += tensor_update
            yield action_list, target_tensor

    def _take_actions(
        self,
        # action_list: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        action_list: List[torch.Tensor],
        target_tensor: torch.Tensor,
    ):
        """Given initial tensor state and action list, update the tensor by
        taking the actions"""
        for action in action_list:
            target_tensor = target_tensor - self._action_to_tensor(action)
        return target_tensor

    def _action_to_tensor(self, action: torch.Tensor):
        factors = (action - 2).split(self.dim_3d, dim=-1)
        return factors_to_tensor(factors)

    def _factor_sample(self):
        distrib = Categorical(self.probs)
        idx_sample = distrib.sample(torch.Size([self.dim_3d]))
        return self.values[idx_sample]


class StrassenDemoDataset(Dataset):
    """Dataset of all valid (states, action) pairs in all permutations of the Strassen factorization
    for 2x2 matrix multiplication.
    Note:
        All data is held in memory (448 demonstrations), not written/read from disk.
        Assumes T=1, does not hold prior tensor states.
    """

    def __init__(self, max_len=None):
        self.n_total = 7
        self.n_demos = 0
        self.state_tensor = []
        self.target_action = []
        self.reward = []
        self.scalar = []
        self.device = "cpu"
        self.bit_info = []
        strassen_tensor, action_list = get_strassen_tensor(self.device)
        uu_strassen, vv_strassen, ww_strassen = get_strassen_factors(self.device)
        for i_bits in range(2**self.n_total):
            bitstring = format(i_bits, "b").zfill(self.n_total)
            used_indexes = [i for i in range(self.n_total) if bitstring[i] == "1"]
            avail_indexes = [i for i in range(self.n_total) if bitstring[i] == "0"]
            # n_used = len(used_indexes)
            n_avail = len(avail_indexes)
            target_tensor = strassen_tensor.clone()
            for j in used_indexes:
                target_tensor -= (
                    uu_strassen[j].view(-1, 1, 1)
                    * vv_strassen[j].view(1, -1, 1)
                    * ww_strassen[j].view(1, 1, -1)
                )

            for k in avail_indexes:
                self.state_tensor.append(target_tensor.unsqueeze(0))
                self.target_action.append(
                    torch.cat((uu_strassen[k], vv_strassen[k], ww_strassen[k])) + 2
                )
                self.reward.append(torch.tensor([-n_avail], dtype=torch.float32))
                self.scalar.append(torch.tensor([0.0], dtype=torch.float32))
                self.bit_info.append(bitstring)
                self.n_demos += 1
        if max_len:
            self.state_tensor = self.state_tensor[:max_len]
            self.target_action = self.target_action[:max_len]
            self.reward = self.reward[:max_len]
            self.scalar = self.scalar[:max_len]
            self.n_demos = max_len

    def __len__(self):
        return self.n_demos

    @torch.no_grad()
    def __getitem__(self, idx: int):
        return (
            self.state_tensor[idx].to(self.device),
            self.scalar[idx].to(self.device),
            self.target_action[idx].to(self.device),
            self.reward[idx].to(self.device),
        )


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


def factors_to_tensor(factors: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
    """Convert the factor triplet defining an action to their outer product tensor."""
    uu, vv, ww = factors
    if uu.dim() == 1:
        tensor_action = uu.view(-1, 1, 1) * vv.view(1, -1, 1) * ww.view(1, 1, -1)
    else:
        tensor_action = (
            uu.unsqueeze(-1).unsqueeze(-1)
            * vv.unsqueeze(-1).unsqueeze(-3)
            * ww.unsqueeze(-2).unsqueeze(-3)
        )
    return tensor_action
