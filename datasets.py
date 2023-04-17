import numpy as np
from pathlib import Path
from typing import List

import torch
from torch.distributions.categorical import Categorical
from torch.utils.data import Dataset

from utils import *

SAVE_DIR_SYNTH_DEMOS = Path("data_unversioned/synthetic_demos")
SAVE_DIR_VAL = Path("data_unversioned/synthetic_demos_val")
SAVE_DIR_PLAYED_GAMES = Path("data_unversioned/played_games")
SAVE_DIR_BEST_GAMES = Path("data_unversioned/best_games")

PLAYED_GAMES_BUFFER_SIZE = 10000
BEST_GAMES_BUFFER_SIZE = 100


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
        shift=1,
        overwrite=True,
        save_dir=SAVE_DIR_SYNTH_DEMOS,
        **kwargs,
    ):
        super().__init__()
        self.max_actions = max_actions
        self.n_demos = n_demos
        self.dim_t = dim_t
        self.dim_3d = dim_3d
        self.values = torch.tensor(values)
        self.probs = torch.tensor(probs)
        self.shift = shift
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        if overwrite:
            for f in self.save_dir.glob("target_tensor_*.pt"):
                f.unlink()
            for f in self.save_dir.glob("action_seq_*.pt"):
                f.unlink()
            n_demos_stored = 0
        else:
            n_demos_stored = len(list(self.save_dir.glob("target_tensor_*.pt")))
        # create demonstrations and save to disk
        if n_demos_stored < n_demos:
            self.n_demos = n_demos_stored
            for i_demo, (action_seq, target_tensor) in enumerate(
                self._create_synthetic_demos(n_demos - n_demos_stored)
            ):
                torch.save(
                    action_seq,
                    self.save_dir.joinpath(f"action_seq_{self.n_demos}.pt"),
                )
                torch.save(
                    target_tensor,
                    self.save_dir.joinpath(f"target_tensor_{self.n_demos}.pt"),
                )
                self.n_demos += 1
        else:
            self.n_demos = n_demos

    def __len__(self):
        return self.n_demos * self.max_actions

    @torch.no_grad()
    def __getitem__(self, idx: int):
        """Returns:
        target_tensor: tensor of shape (dim_t, dim_3d, dim_3d, dim_3d)
        scalar: scalar of shape (1)
        action: action of shape (12)
        reward: reward of shape (1)"""
        idx_demo = idx // self.max_actions
        idx_action = idx % self.max_actions
        action_seq = torch.load(self.save_dir.joinpath(f"action_seq_{idx_demo}.pt"))
        target_tensor = torch.load(
            self.save_dir.joinpath(f"target_tensor_{idx_demo}.pt"),
        )
        if idx_action != self.max_actions - 1:
            actions = action_seq[idx_action + 1 :]
            target_tensor = self._take_actions(actions, target_tensor)
        action = action_seq[idx_action]
        target_tensor = torch.stack(
            [
                target_tensor,
                *(
                    action_to_tensor(action)
                    for action in reversed(
                        action_seq[idx_action + 1 : idx_action + self.dim_t]
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
        reward = torch.tensor([-(idx_action + 1)], dtype=torch.float32)
        return (
            target_tensor.to(self.device),
            scalar.to(self.device),
            action.to(self.device),
            reward.to(self.device),
        )

    def _create_synthetic_demos(self, n_demos_needed: int):
        """Generate a mult tensor and a list of actions producing it"""
        for _ in range(n_demos_needed):
            target_tensor = torch.zeros(self.dim_3d, self.dim_3d, self.dim_3d)
            action_seq = []
            for i in range(self.max_actions):
                valid_action = False
                while not valid_action:
                    uu = self._factor_sample()
                    vv = self._factor_sample()
                    ww = self._factor_sample()

                    tensor_action = uvw_to_tensor((uu, vv, ww))

                    if not (tensor_action == 0).all():
                        valid_action = True
                        action_seq.append(torch.cat((uu, vv, ww)) + self.shift)
                        target_tensor += tensor_action
            yield action_seq, target_tensor

    @staticmethod
    def _take_actions(
        action_seq: List[torch.Tensor],
        target_tensor: torch.Tensor,
    ):
        """Given initial tensor state and action list, update the tensor by
        taking the actions"""
        for action in action_seq:
            target_tensor = target_tensor - action_to_tensor(action)
        return target_tensor

    def _factor_sample(self):
        distrib = Categorical(self.probs)
        idx_sample = distrib.sample(torch.Size([self.dim_3d]))
        return self.values[idx_sample]


class PlayedGamesDataset(Dataset):
    """Create a dataset of played games."""

    def __init__(
        self,
        buffer_size: int,
        device: str,
        save_dir=SAVE_DIR_PLAYED_GAMES,
        **kwargs,
    ):
        super().__init__()
        self.game_pointer = 0
        self.buffer_size = buffer_size
        self.game_lengths = {}
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def __del__(self):
        """Delete all saved games."""
        for f in self.save_dir.glob("*.pt"):
            f.unlink()
        self.game_pointer = 0

    def __len__(self):
        return sum(self.game_lengths.values())

    @torch.no_grad()
    def __getitem__(self, idx: int):
        """Get a game from the dataset.
        Returns:
            target_tensor: tensor of shape (dim_t, dim_3d, dim_3d, dim_3d)
            scalar: scalar of shape (1)
            action: action of shape (12)
            reward: reward of shape (1)"""
        i = 0
        while idx >= self.game_lengths[i]:
            idx -= self.game_lengths[i]
            i += 1
        state_seq = torch.load(Path(self.save_dir, f"state_seq_{i}.pt"))
        action_seq = torch.load(Path(self.save_dir, f"action_seq_{i}.pt"))
        reward_seq = torch.load(Path(self.save_dir, f"reward_seq_{i}.pt"))
        return (
            state_seq[idx].to(self.device),
            get_scalars(state_seq[idx], idx, batch_size=False).to(self.device),
            action_seq[idx].to(self.device).argmax(dim=-1),
            # Why do this over the prob dist?
            # action_seq[idx].to(self.device).argmax(dim=-1),
            reward_seq[idx].reshape(1).to(self.device),
        )

    def add_game(
        self,
        state_seq: List[torch.Tensor],
        action_seq: List[torch.Tensor],
        reward_seq: List[torch.Tensor],
    ):
        """Add game to current location of game_pointer and increment pointer."""
        self.game_lengths[self.game_pointer] = len(state_seq)
        torch.save(
            state_seq,
            self.save_dir.joinpath(f"state_seq_{self.game_pointer}.pt"),
        )
        torch.save(
            action_seq,
            self.save_dir.joinpath(f"action_seq_{self.game_pointer}.pt"),
        )
        torch.save(
            reward_seq,
            self.save_dir.joinpath(f"reward_seq_{self.game_pointer}.pt"),
        )
        self.game_pointer = (self.game_pointer + 1) % self.buffer_size


class TensorGameDataset(Dataset):
    def __init__(
        self,
        len_data: int,
        fract_synth: float,
        max_actions: int,
        dim_t: int,
        dim_3d: int,
        device: str,
        start_tensor=None,
        **kwargs,
    ):
        super().__init__()
        self.len_data = len_data
        self.buffer_synth = SyntheticDemoDataset(
            max_actions,
            len_data,
            dim_t,
            dim_3d,
            device,
            **kwargs,
        )
        self.buffer_played = PlayedGamesDataset(
            PLAYED_GAMES_BUFFER_SIZE, device, save_dir=SAVE_DIR_PLAYED_GAMES
        )
        self.buffer_best = PlayedGamesDataset(
            BEST_GAMES_BUFFER_SIZE, device, save_dir=SAVE_DIR_BEST_GAMES
        )
        self.is_synth = torch.ones(len_data, dtype=torch.bool)
        self.index_synth = torch.from_numpy(
            np.random.choice(len(self.buffer_synth), len_data, replace=False)
        )
        self.index_played = None
        self.index_best = None
        self.fract_synth = fract_synth
        self.fract_best = 0
        self.dim_t = dim_t
        self.dim_3d = dim_3d
        self.device = device
        matrix_size = int(np.sqrt(dim_3d))
        if start_tensor is None:
            self.start_tensor = build_matmul_tensor(
                dim_t, matrix_size, matrix_size, matrix_size
            )
        else:
            self.start_tensor = start_tensor

    def __len__(self):
        """Return the length of the dataset."""
        return self.len_data

    def __getitem__(self, idx: int):
        """Get a state from the dataset, from one of buffer_synth, buffer_played, buffer_best."""
        len_synth = self.is_synth[:idx].sum()
        if self.is_synth[idx]:
            return self.buffer_synth[int(self.index_synth[len_synth])]
        else:
            synth_remainder = idx - len_synth
            if self.fract_best > 0 and self.index_best is not None:
                len_best = len(self.index_best)
                if synth_remainder < len_best:
                    return self.buffer_best[int(self.index_best[synth_remainder])]
                else:
                    synth_best_remainder = synth_remainder - len_best
                    return self.buffer_played[
                        int(self.index_played[synth_best_remainder])
                    ]
            else:
                return self.buffer_played[int(self.index_played[synth_remainder])]

    def set_fractions(self, fract_synth, fract_best):
        self.fract_synth = fract_synth
        self.fract_best = fract_best

    def resample_buffer_indexes(self):
        """If played games buffer is not empty, do a random draw to
        reset the indexes to synth/played/best buffers."""
        if len(self.buffer_played) > 0:
            # set new indexes for synthetic demos
            self.is_synth = torch.rand(self.len_data) < self.fract_synth
            len_synth = self.is_synth.sum().item()
            self.index_synth = torch.from_numpy(
                np.random.choice(len(self.buffer_synth), len_synth, replace=False)
            )
            if len(self.buffer_best) > 0 and self.fract_best > 0:
                # set new indexes for played games and for best games
                len_played = int(1 - self.fract_synth - self.fract_best) * self.len_data
                len_best = self.len_data - len_synth - len_played
                replace_played = len_played > len(self.buffer_played)
                replace_best = len_best > len(self.buffer_best)
                self.index_played = torch.from_numpy(
                    np.random.choice(
                        len(self.buffer_played), len_played, replace=replace_played
                    )
                )
                self.index_best = torch.from_numpy(
                    np.random.choice(
                        len(self.buffer_best), len_best, replace=replace_best
                    )
                )
            else:
                # set new indexes for played games only (if no best games)
                len_played = self.len_data - len_synth
                replace_played = len_played > len(self.buffer_played)
                self.index_played = torch.from_numpy(
                    np.random.choice(
                        len(self.buffer_played), len_played, replace=replace_played
                    )
                )

    def add_played_game(
        self,
        state_seq: List[torch.Tensor],
        action_seq: List[torch.Tensor],
        reward_seq: List[torch.Tensor],
    ):
        self.buffer_played.add_game(state_seq, action_seq, reward_seq)

    def add_best_game(
        self,
        state_seq: List[torch.Tensor],
        action_seq: List[torch.Tensor],
        reward_seq: List[torch.Tensor],
    ):
        self.buffer_best.add_game(state_seq, action_seq, reward_seq)

    # @property
    # def target_tensor(self) -> torch.Tensor:
    #     max_matrix_size = int(np.sqrt(self.dim_3d))
    #     initial_state = torch.zeros(
    #         1,
    #         self.dim_t,
    #         self.dim_3d,
    #         self.dim_3d,
    #         self.dim_3d,
    #     )
    #     # matrix_dims = (
    #     #     torch.randint(1, max_matrix_size, (3,)).detach().cpu().numpy().tolist()
    #     # )
    #     # operation_tensor = self._build_initial_state(*matrix_dims, self.dim_t)
    #     operation_tensor = build_matmul_tensor(max_matrix_size, max_matrix_size, max_matrix_size, self.dim_t)
    #     # operation_tensor = self._build_initial_state(max_matrix_size, max_matrix_size, max_matrix_size, self.dim_t)
    #     initial_state[
    #         0,
    #         :,
    #         : operation_tensor.shape[1],
    #         : operation_tensor.shape[2],
    #         : operation_tensor.shape[3],
    #     ] = operation_tensor
    #     return initial_state.to(self.device)
    #
    # @staticmethod
    # def _build_initial_state(dim_1: int, dim_k: int, dim_2: int, dim_t: int):
    #     """Build the initial state for the game/act step. The input tensor has shape
    #     (dim_t, dim_3d, dim_3d, dim_3d).
    #     The first slice represent the matrix multiplication tensor which will
    #     be reduced by the TensorGame algorithm. The other slices represent the
    #     previous tensor state memory and are set to zero for the initial state.
    #     """
    #     initial_state = torch.zeros(
    #         dim_t, dim_1 * dim_k, dim_k * dim_2, dim_1 * dim_2
    #     )
    #     for ij in range(dim_1 * dim_2):
    #         for k in range(dim_k):
    #             initial_state[
    #                 0, (ij // dim_2) * dim_k + k, k * dim_2 + ij % dim_2, ij
    #             ] = 1
    #     return initial_state


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
        strassen_tensor, action_seq = get_strassen_tensor(self.device)
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
    return uvw_to_demo(uu_strassen, vv_strassen, ww_strassen, device)
