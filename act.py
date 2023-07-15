from typing import Dict, List
import torch

from model import AlphaTensor
from utils import *


def actor_prediction(
    model: AlphaTensor,
    initial_state: torch.Tensor,
    max_actions: int,
    n_sim: int,
    n_bar: int,
):
    """Given a model and initial state, produce a single game trajectory.

    Args:
        model: AlphaTensor model to select actions
        initial_state: Initial state of trajectory.
        max_actions: The maximum number of actions to take in a trajectory
        n_sim: The number of simulations to run.
        n_bar: The parameter used to compute the improved policy.
    Returns:
        state_seq: A list of states in the trajectory.
        action_seq: A list of actions in the trajectory.
        reward_seq: A list of rewards in the trajectory.
    """
    state = initial_state.unsqueeze(0)
    state_seq = []
    mc_tree = {}
    state_info = {}
    string_seq = []
    i_action = 0
    while i_action < max_actions:
        state_seq.append(state)
        headstate = get_head_state(state)
        state_string = state_to_str(headstate)
        string_seq.append(state_string)
        # call mc_ts at each timestep to select next (action, state)
        state, mc_tree, state_info = mc_ts(
            model,
            state,
            n_sim,
            i_action,
            max_actions,
            mc_tree,
            state_info,
        )
        if tensor_factorized(state):
            break
        i_action += 1
    policy_seq = get_improved_policy(
        state_info,
        string_seq,
        model.n_steps,
        model.n_logits,
        n_bar,
    )
    end_state_reward = -get_rank(state)
    reward_seq = torch.cumsum(
        torch.tensor([-1] * (len(policy_seq) - 1) + [-1 + end_state_reward]), dim=0
    )
    state_seq = [s.squeeze(0) for s in state_seq]
    return state_seq, policy_seq, reward_seq


def mc_ts(
    model: AlphaTensor,
    root_state: torch.Tensor,
    n_sim: int,
    i_action: int,
    max_actions: int,
    mc_tree: dict,
    state_info: dict,
):
    """Monte Carlo tree search.

    Args:
        model: main AlphaTensor model.
        root_state: Root state for next step exploration.
        n_sim: The number of simulated paths to play from root state.
        i_action: The current action index.
        max_actions: The maximum number of actions to play to.
        mc_tree: Tree of states explored by MC.
        state_info: Stores auxiliary information for each state in mc_tree.

    Returns:
        next_state (torch.Tensor): Next state based on action chosen by MCTS
        mc_tree: Tree updated based on chosen action
        state_info: Info updated based on chosen action
    """
    head_state = get_head_state(root_state)
    state_string = state_to_str(head_state)
    if state_string in state_info:
        with torch.no_grad():
            visit_count = state_info[state_string][3]
            # reduce number of sims to run by number of previous visits
            n_sim -= int(visit_count.sum())
            n_sim = max(n_sim, 0)

    for i_mc in range(n_sim):
        mc_tree, state_info = extend_tree(
            model, root_state, i_action, max_actions, mc_tree, state_info
        )

    candidate_states, _, _, visit_count, q_vals, _ = state_info[state_string]
    # candidate_states = expand_candidate_states(candidate_states)
    repetitions = {i: [] for i, _ in enumerate(candidate_states)}
    next_state_idx = select_next_state(
        candidate_states, q_vals, visit_count, repetitions, return_idx=True
    )
    next_state = candidate_states[next_state_idx]
    return next_state, mc_tree, state_info


@torch.no_grad()
def extend_tree(
    model: AlphaTensor,
    state: torch.Tensor,
    i_action: int,
    max_actions: int,
    mc_tree: Dict,
    state_info: Dict,
    horizon=5,
):
    """A single branching exploration from a root state. Follow root state to end of
    current MC tree and extend one step by sampling from model actions.

    Args:
        model: main AlphaTensor model.
        state: Root state to explore from.
        i_action: The current action index.
        max_actions: The maximum number of actions to play forward to.
        mc_tree: Tree of states explored by MC.
        state_info: Dict of auxiliary information for each state in mc_tree.
        horizon: Number of actions to play forward from root state.

    Returns:
        new_state_info
        new_mc_tree
    """
    new_state_info = state_info.copy()
    new_mc_tree = mc_tree.copy()
    idx = i_action
    max_actions_mc = min(max_actions, i_action + horizon)
    head_state = get_head_state(state)
    state_string = state_to_str(head_state)
    trajectory = []
    while state_string in new_mc_tree:
        (
            candidate_states,
            _,
            _,
            visit_count,
            q_vals,
            action_seq,
        ) = new_state_info[state_string]
        # candidate_states = expand_candidate_states(candidate_states_dict)
        repetitions = {i: [] for i, _ in enumerate(candidate_states)}
        state_idx = select_next_state(
            candidate_states,
            q_vals,
            visit_count,
            repetitions,
            # rep_map,
            return_idx=True,
        )
        trajectory.append((state_string, state_idx))
        if len(trajectory) > 2 * max_actions:
            print("trajectory too long")
            pass
        state = candidate_states[state_idx]
        head_state = get_head_state(state)
        state_string = state_to_str(head_state)
        idx += 1

    # extension
    if idx <= max_actions_mc:
        trajectory.append((state_string, None))
        if not tensor_factorized(get_head_state(state)):
            state = state.to(model.device)
            scalars = get_scalars(state, idx).to(model.device)
            # scalars = scalars_local.to(model.device)
            candidate_states = []
            while len(candidate_states) == 0:
                actions, _, q_vals = model.fwd_infer(state, scalars)
                candidate_states = get_child_states(state, actions)
                # prune candidates which repeat previous states
                idxs = remove_null_actions(state, candidate_states)
                actions = actions[:, idxs]
                candidate_states = [candidate_states[i] for i in idxs]
                candidate_strings = [
                    state_to_str(get_head_state(c)) for c in candidate_states
                ]
                idxs = [
                    i for i, c in enumerate(candidate_strings) if c not in new_mc_tree
                ]
                actions = actions[:, idxs]
                candidate_states = [candidate_states[i] for i in idxs]
            not_dupl_actions = actions[:, :].to("cpu")
            not_dupl_q_values = torch.zeros(not_dupl_actions.shape[:-1]).to("cpu")
            visit_count = torch.zeros_like(not_dupl_q_values).to("cpu")
            head_state = get_head_state(state)
            state_string = state_to_str(head_state)
            new_state_info[state_string] = (
                candidate_states,
                0,  # cloned_idx_to_idx
                0,  # repetitions
                visit_count,
                not_dupl_q_values,
                not_dupl_actions,
            )
            new_mc_tree[state_string] = [
                state_to_str(get_head_state(c)) for c in candidate_states
            ]
            leaf_q_val = q_vals
    else:
        leaf_q_val = -get_rank(state)
    new_state_info = backward_pass(trajectory, new_state_info, leaf_q_val)
    return new_mc_tree, new_state_info


def backward_pass(trajectory, state_info, leaf_q_val):
    new_state_info = state_info.copy()
    reward = 0
    for idx, (state, action_idx) in enumerate(reversed(trajectory)):
        if action_idx is None:
            reward += leaf_q_val
        else:
            # visit_count = new_state_info[state][3]
            # q_vals = new_state_info[state][4]
            if isinstance(reward, torch.Tensor):
                reward = reward.to(new_state_info[state][4].device)
            action_idx = int(action_idx)
            # TO DO add dupl lines here
            not_dupl_index = action_idx
            # TO DO : is this wrong sign?
            reward -= 1
            new_state_info[state][4][:, not_dupl_index] = (
                new_state_info[state][3][:, not_dupl_index]
                * new_state_info[state][4][:, not_dupl_index]
                + reward
            ) / (new_state_info[state][3][:, not_dupl_index] + 1)
            new_state_info[state][3][:, not_dupl_index] += 1
            # new_state_info[state][3] = visit_count
            # new_state_info[state][4] = q_vals
    return new_state_info


def select_next_state(
    candidate_states: List[torch.Tensor],
    q_vals: torch.Tensor,
    visit_count: torch.Tensor,
    reps: Dict[int, list],
    c1=1.25,
    c2=19652.0,
    return_idx: bool = False,
):
    """Select the next state, from among candidates, which maximizes UCB"""
    pi = torch.tensor(
        [len(reps[i]) for i in range(len(candidate_states)) if i in reps]
    ).to(q_vals.device)
    if pi.shape[0] != visit_count.shape[1]:
        # print(pi)
        # print(pi.shape, q_vals.shape, visit_count.shape)
        pi = pi[: visit_count.shape[1]]
    sum_visits = visit_count.sum()
    c_explore = c1 + torch.log((sum_visits + c2 + 1) / c2)
    # ucb = (
    #     q_vals.reshape(-1) + pi * torch.sqrt(sum_visits / (1 + visit_count)) * c_explore
    # )
    ucb = q_vals.reshape(-1) + c_explore * pi * torch.sqrt(sum_visits) / (
        1 + visit_count
    )
    if return_idx:
        return ucb.argmax()
    else:
        return candidate_states[ucb.argmax()]


def get_child_states(state: torch.Tensor, actions: torch.Tensor, vec_cardinality=5):
    bs, k, n_steps = actions.shape[:3]
    action_tensor = action_to_tensor(actions)
    # TO DO: remove duplicates if nec
    # action_tensor, _ = remove_duplicates(action_tensor)
    initial_head_state = get_head_state(state)
    new_head_states = initial_head_state - action_tensor
    new_states = [
        torch.cat([new_head_states[:, i : i + 1], state[:, :-1]], dim=1)
        for i in range(k)
    ]
    return new_states
    # rolling_states = torch.roll(state, 1)[:, 2:]
    # return [
    #     torch.cat(
    #         [
    #             new_head_states[:, i : i + 1],
    #             action_tensor[:, i : i + 1],
    #             rolling_states,
    #         ],
    #         dim=1,
    #     )
    #     for i in range(k)
    # ]


@torch.no_grad()
def get_improved_policy(
    state_info: Dict,
    string_seq: List[str],
    n_steps: int,
    n_logits: int,
    n_bar: int,
):
    policy_seq = torch.zeros(len(string_seq), n_steps, n_logits)
    n_bar = torch.tensor(n_bar)
    for ii, state_str in enumerate(string_seq):
        _, _, _, visit_count, _, action_cands = state_info[state_str]
        sum_visits = visit_count.sum()
        if sum_visits > n_bar:
            tau = (sum_visits.log() / n_bar.log()).item()
        else:
            tau = 1
        visit_count = visit_count ** (1 / tau)
        improved_policy = visit_count / sum_visits
        for sample_id in range(action_cands.shape[1]):
            action_ids = action_cands[0, sample_id]
            for step_id, action_id in enumerate(action_ids):
                policy_seq[ii, step_id, action_id] += improved_policy[0, sample_id]
    return policy_seq


# def actor_prediction_simple(
#     model: AlphaTensor,
#     initial_state: torch.Tensor,
#     max_actions: int,
# ):
#     """Given a model and initial state, produce a game trajectory."""
#     state = initial_state.unsqueeze(0)
#     state_seq = []
#     action_seq = []
#     reward_seq = []
#     i_action = 0
#     while i_action < max_actions:
#         scalars = get_scalars(state, i_action)
#         aa, pp, qq = model.fwd_infer(state, scalars)
#         # aa, pp, qq = model.fwd_infer(state, scalars, n_samples=1)
#         # action_tensor = action_to_tensor(aa.squeeze())
#         ## while not torch.all(torch.eq(action_tensor, 0)):
#         # while not (aa==0).all():
#         #     aa, pp, qq = model.fwd_infer(state, scalars, n_samples=1)
#         #     action_tensor = action_to_tensor(aa.squeeze())
#         #     print(aa)
#         state = update_state(state, aa, batch=True)
#         action_seq.append(aa[0, 0])
#         state_seq.append(state[0])
#         i_action += 1
#         if get_rank(state) == 0:
#             # reward = torch.tensor([0])
#             # reward_seq.append(reward)
#             break
#         # elif i_action == max_actions:
#         #     reward = -get_rank(state)
#         # else:
#         #     reward = torch.tensor([-1])
#         # reward_seq.append(reward)
#     # TO DO: merge this with get_rank
#     # end_state_reward0 = -int(torch.linalg.matrix_rank(state[0, 0]).sum())
#     end_state_reward = -get_rank(state)
#     reward_seq = torch.cumsum(
#         torch.tensor([-1] * (len(action_seq) - 1) + [-1 + end_state_reward]), dim=0
#     )
#     return state_seq, action_seq, reward_seq
