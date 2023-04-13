from typing import Dict, List
import torch

from model import AlphaTensor
from utils import *


@torch.no_grad()
def simulate_game(
    model: AlphaTensor,
    state: torch.Tensor,
    i_action: int,
    max_actions: int,
    mc_tree: Dict,
    state_info: Dict,
    horizon=5,
):
    """A single branching exploration from root state.

    Args:
        model: main AlphaTensor model.
        state: Root state to play from.
        i_action: The current action index.
        max_actions: The maximum number of actions to play to.
        mc_tree: Tree of states explored by MC.
        state_info: Dict of auxiliary information for each state in mc_tree.
        horizon: Number of actions to play forward from root state.

    Returns:
        state_info
        mc_tree
    """
    idx = i_action
    max_steps = min(max_actions, i_action + horizon)
    state_string = state_to_str(get_current_state(state))
    trajectory = []
    while state_string in mc_tree:
        (
            # candidate_states_dict,
            candidate_states,
            # old_idx_to_new_idx,
            # rep_map,
            visit_count,
            q_vals,
            action_seq,
        ) = state_info[state_string]
        # candidate_states = expand_candidate_states(candidate_states_dict)
        state_idx = select_next_state(
            candidate_states,
            q_vals,
            visit_count,
            # rep_map,
            return_idx=True,
        )
        trajectory.append((state_string, state_idx))
        state = candidate_states[state_idx]
        next_state = get_current_state(state)
        state_string = state_to_str(next_state)
        idx += 1

    # expansion
    if idx <= max_steps:
        trajectory.append((state_string, None))
        if not tensor_factorized(get_current_state(state)):
            state = state.to(model.device)
            scalars = get_scalars(state, idx).to(state.device)
            actions, probs, q_vals = model.fwd_infer(state, scalars, n_samples=1)
            candidate_states = get_child_states(state, actions)
            not_dupl_actions = actions[:, :].to("cpu")
            not_dupl_q_values = torch.zeros(not_dupl_actions.shape[:-1]).to("cpu")
            visit_count = torch.zeros_like(not_dupl_q_values).to("cpu")
            current_state = get_current_state(state)
            state_info[state_to_str(current_state)] = (
                candidate_states,
                # reps,
                visit_count,
                not_dupl_q_values,
                not_dupl_actions,
            )
            mc_tree[state_to_str(current_state)] = [
                state_to_str(get_current_state(candidate))
                for candidate in candidate_states
            ]
            leaf_q_val = q_vals
    else:
        leaf_q_val = -int(torch.linalg.matrix_rank(state).sum())
    backward_pass(trajectory, state_info, leaf_q_val)


def backward_pass(trajectory, state_info, leaf_q_val):
    pass

    #
    # state = initial_state
    # time_step = initial_time
    # state_list = [state[:, 0]]
    # action_list = []
    #
    # while time_step < max_steps:
    #     state = state.to(model.device)
    #     actions, probs, q_vals = model.fwd_infer(state, time_step)
    #     action_list.append(actions)
    #     # state =
    #     # state_list
    #     # time_step += 1


def actor_prediction(
    model: AlphaTensor,
    initial_state: torch.Tensor,
    max_actions: int,
):
    """Given a model and initial state, produce a game trajectory."""
    state = initial_state.unsqueeze(0)
    state_seq = []
    action_seq = []
    reward_seq = []
    i_action = 0
    while i_action < max_actions:
        scalars = get_scalars(state, i_action)
        aa, pp, qq = model.fwd_infer(state, scalars, n_samples=1)
        # action_tensor = action_to_tensor(aa.squeeze())
        ## while not torch.all(torch.eq(action_tensor, 0)):
        # while not (aa==0).all():
        #     aa, pp, qq = model.fwd_infer(state, scalars, n_samples=1)
        #     action_tensor = action_to_tensor(aa.squeeze())
        #     print(aa)
        state = update_state(state, aa, batch=True)
        action_seq.append(aa[0, 0])
        state_seq.append(state[0])
        i_action += 1
        if get_rank(state) == 0:
            # reward = torch.tensor([0])
            # reward_seq.append(reward)
            break
        # elif i_action == max_actions:
        #     reward = -get_rank(state)
        # else:
        #     reward = torch.tensor([-1])
        # reward_seq.append(reward)
    # TO DO: merge this with get_rank
    # end_state_reward0 = -int(torch.linalg.matrix_rank(state[0, 0]).sum())
    end_state_reward = -get_rank(state)
    reward_seq = torch.cumsum(
        torch.tensor([-1] * (len(action_seq) - 1) + [-1 + end_state_reward]), dim=0
    )
    return state_seq, action_seq, reward_seq


def actor_prediction_mcts(
    model: AlphaTensor,
    initial_state: torch.Tensor,
    max_actions: int,
    n_mc: int,
    n_bar: int,
):
    """Given a model and initial state, produce a game trajectory."""
    state = initial_state.unsqueeze(0)
    state_seq = []
    mc_tree = {}
    state_info = {}
    state_strings = []
    i_action = 0
    while i_action < max_actions:
        state_seq.append(state)
        state_strings.append(state_to_str(get_current_state(state)))
        state = mc_ts(
            model,
            state,
            n_mc,
            i_action,
            max_actions,
            mc_tree,
            state_info,
        )
        if tensor_factorized(state):
            break
        i_action += 1
    end_state = get_current_state(state)
    action_seq = get_improved_policy(
        state_info,
        state_strings,
        model.n_steps,
        model.n_logits,
        n_bar,
    )
    end_state_reward = -get_rank(state)
    reward_seq = torch.cumsum(
        torch.tensor([-1] * (len(action_seq) - 1) + [-1 + end_state_reward]), dim=0
    )
    state_seq = [s.squeeze(0) for s in state_seq]
    return state_seq, action_seq, reward_seq


def mc_ts(
    model: AlphaTensor,
    state: torch.Tensor,
    n_mc: int,
    i_action: int,
    n_actions: int,
    mc_tree: dict,
    state_info: dict,
):
    """Monte Carlo tree search.

    Args:
        model (torch.nn.Module): main AlphaTensor model.
        state (torch.Tensor): Root state for MC tree.
        n_mc (int): The number of simulated games to play.
        i_action (int): The current action index.
        n_actions (int): The maximum number of actions to play to.
        mc_tree (Dict): Tree of states explored by MC.
        state_info (Dict): Stores auxiliary information for each state in mc_tree.

    Returns:
        next_state (torch.Tensor): Next state based on action chosen by MCTS.
    """
    state_string = state_to_str(get_current_state(state))
    if state_string in state_info:
        with torch.no_grad():
            visit_count = state_info[state_string][3]
            n_mc -= int(visit_count.sum())
            n_mc = max(n_mc, 0)

    for _ in range(n_mc):
        simulate_game(model, state, i_action, n_actions, mc_tree, state_info)

    candidate_states, _, visit_count, q_vals, _ = state_info[state_string]
    candidate_states = expand_candidate_states(candidate_states)
    next_state_idx = select_next_state(
        candidate_states, q_vals, visit_count, return_idx=True
    )
    next_state = candidate_states[next_state_idx]
    return next_state


def expand_candidate_states(candidate_states: Dict):
    final_states = candidate_states["final_states"]
    previous_actions = candidate_states["previous_actions"]
    full_candidate_states = [
        torch.cat([final_states[i], previous_actions], dim=1)
        for i in range(len(final_states))
    ]
    return full_candidate_states


def select_next_state(
    candidate_states: List[torch.Tensor],
    q_vals: torch.Tensor,
    visit_count: torch.Tensor,
    reps: Dict[int, list],
    c1=1.25,
    c2=19652.0,
    return_idx: bool = False,
):
    """Select next state from candidates that maximizes UCB"""
    pi = torch.tensor(
        [len(reps[i]) for i in range(len(candidate_states)) if i in reps]
    ).to(q_vals.device)
    if pi.shape[0] != visit_count[1]:
        print(pi)
        print(pi.shape, q_vals.shape, visit_count.shape)
        pi = pi[: visit_count[1]]
    # ucb = q_vals.reshape(-1) + (
    #     c1 + torch.log((torch.sum(visit_count) + c2 + 1) / c2)
    # ) * pi * torch.sqrt(torch.sum(visit_count)) / (1 + visit_count)
    ucb = q_vals.reshape(-1) + pi * torch.sqrt(
        torch.sum(visit_count) / (1 + visit_count)
    ) * (c1 + torch.log((torch.sum(visit_count) + c2 + 1) / c2))
    if return_idx:
        return ucb.argmax()
    else:
        return candidate_states[ucb.argmax()]


def get_child_states(state: torch.Tensor, actions: torch.Tensor, vec_cardinality=5):
    bs, k, n_steps = actions.shape[:3]
    action_tensor = action_to_tensor(actions)
    # TO DO: remove duplicates if nec
    # action_tensor, _ = remove_duplicates(action_tensor)
    initial_state = get_current_state(state)
    new_state = initial_state - action_tensor
    rolling_states = torch.roll(state, 1)[:, 2:]
    return [
        torch.cat(
            [
                new_state[:, i : i + 1],
                action_tensor[:, i : i + 1],
                rolling_states,
            ],
            dim=1,
        )
        for i in range(k)
    ]


@torch.no_grad()
def get_improved_policy(
    state_info: Dict,
    state_strings: List[str],
    n_steps: int,
    n_logits: int,
    n_bar: int,
):
    policies = torch.zeros(len(state_strings), n_steps, n_logits)
    n_bar = torch.tensor(n_bar)
    for ii, state in enumerate(state_strings):
        _, _, _, visit_count, _, action_seq = state_info[state]
        if visit_count.sum() > n_bar:
            tau = (torch.log(visit_count.sum()) / torch.log(n_bar)).item()
        else:
            tau = 1
        visit_count = visit_count ** (1 / tau)
        improved_policy = visit_count / visit_count.sum()
        for sample_id in range(action_seq.shape[1]):
            action_ids = action_seq[0, sample_id]
            for step_id, action_id in enumerate(action_ids):
                policies[ii, step_id, action_id] += improved_policy[0, sample_id]
    return policies
