import os.path
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

SAVE_DIR_SYNTH_DEMOS = str(Path.home() / "./data/synthetic_demos")


def factors_to_tensor(factors: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
    """ convert the factors defining an action to their outer product"""
    uu, vv, ww = factors
    tensor_action = uu.view(-1, 1, 1) * vv.view(1, -1, 1) * ww.view(1, 1, -1)
    return tensor_action


def take_actions(
    action_list: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    target_tensor: torch.Tensor,
):
    """Given initial tensor state and action list, update the tensor by
    taking the actions"""
    for action in action_list:
        target_tensor = target_tensor - factors_to_tensor(action)
    return target_tensor


class SyntheticDemoBuffer(Dataset):
    """Create a set of synthetic demonstrations and save to disk."""

    def __init__(
        self,
        max_rank: int,
        n_demos: int,
        dim_t: int,
        dim_3d: int,
        distrib: torch.distributions.categorical.Categorical,
        device,
        save_dir=SAVE_DIR_SYNTH_DEMOS,
    ):
        self.max_rank = max_rank
        self.n_demos = n_demos
        self.dim_t = dim_t
        self.dim_3d = dim_3d
        self.distrib = distrib
        self.device = device
        self.save_dir = save_dir
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        n_demos_stored = len(list(Path(self.save_dir).glob("target_tensor_*.pt")))
        if n_demos_stored < n_demos:
            self.len_data = n_demos_stored
            for i_demo, (action_list, target_tensor) in enumerate(
                self._create_synthetic_demos(n_demos - n_demos_stored)
            ):
                torch.save(
                    action_list,
                    os.path.join(self.save_dir, f"action_list_{self.len_data}.pt"),
                )
                torch.save(
                    target_tensor,
                    os.path.join(self.save_dir, f"target_tensor_{self.len_data}.pt"),
                )
            self.len_data += 1
        else:
            self.len_data = n_demos

    def __len__(self):
        return self.len_data * self.max_rank

    @torch.no_grad()
    def __getitem__(self, idx: int):
        i = idx // self.max_rank
        j = idx % self.max_rank
        action_list = torch.load(
            os.path.join(self.save_dir, f"action_list_{self.len_data}.pt")
        )
        target_tensor = torch.load(
            os.path.join(self.save_dir, f"target_tensor_{self.len_data}.pt"),
        )
        if j != self.max_rank - 1:
            actions = action_list[j + 1 :]
            target_tensor = take_actions(actions, target_tensor)
        action = action_list[j]
        target_tensor = torch.stack(
            [
                target_tensor,
                *(
                    factors_to_tensor(t)
                    for t in reversed(action_list[j + 1 : j + self.dim_t])
                ),
            ]
        )
        if len(target_tensor) < self.dim_t:
            target_tensor = torch.cat(
                [
                    target_tensor,
                    torch.zeros(
                        self.dim_t - len(target_tensor), *target_tensor.shape[1:],
                    ),
                ]
            )
        scalar = torch.tensor(self.max_rank - j).unsqueeze(-1).float()
        policy = torch.cat(action)
        reward = torch.tensor([-(j + 1)])
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
            for i in range(self.max_rank):
                valid_action = False
                while not valid_action:
                    uu = self.distrib(self.dim_3d)
                    vv = self.distrib(self.dim_3d)
                    ww = self.distrib(self.dim_3d)
                    tensor_update = (
                        uu.view(-1, 1, 1) * vv.view(1, -1, 1) * ww.view(1, 1, -1)
                    )
                    if not (tensor_update == 0).all():
                        valid_action = True
                        action_list.append((uu, vv, ww))
                        target_tensor += tensor_update
            yield action_list, target_tensor
