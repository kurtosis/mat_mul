import torch

from model import AlphaTensor
from utils import *


@torch.no_grad()
def simulate_game(
    model, initial_state: torch.Tensor, initial_time: int, max_steps: int
):
    """NOT USING YET. Plays one game simulation from a given state"""
    scalar = initial_time
    trajectory = []
    state = initial_state
    time_step = initial_time
    state_list = [state[:, 0]]
    action_list = []

    while time_step < max_steps:
        state = state.to(model.device)
        actions, probs, q_vals = model.fwd_infer(state, time_step)
        action_list.append(actions)
        # state =
        # state_list
        # time_step += 1


def actor_prediction(
    model: AlphaTensor,
    initial_state: torch.Tensor,
    max_actions: int,
):
    """Given a model and initial state, produce a game trajectory."""
    i_action = 0
    state = initial_state.unsqueeze(0)
    state_seq = []
    action_seq = []
    reward_seq = []
    while i_action < max_actions:
        scalars = get_scalars(state, i_action)
        aa, pp, qq = model.fwd_infer(state, scalars, n_samples=1)
        state = update_state(state, aa, batch=True)
        action_seq.append(aa[0, 0])
        state_seq.append(state[0])
        i_action += 1
        if get_rank(state) == 0:
            reward = torch.tensor([0])
            reward_seq.append(reward)
            break
        elif i_action == max_actions:
            reward = get_rank(state)
            reward_seq.append(reward)
        else:
            reward = torch.tensor([-1])
            reward_seq.append(reward)
    return state_seq, action_seq, reward_seq
