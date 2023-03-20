import torch

from data_generation import *
from model import *
from utils import *

device = "cpu"
lr = 1e-3
n_epochs = 20000
weight_pol = 1
weigh_val = 1
n_print = 100
max_len = None
# check_interval = 5

alpha = AlphaTensor(
    dim_3d=4,
    dim_t=1,
    dim_s=1,
    dim_c=16,
    n_samples=1,
    n_steps=12,
    n_logits=4,
    n_feats=8,
    n_heads=4,
    n_hidden=8,
    # drop_p=0.1,
    device=device,
)
alpha = alpha.to(device)
optimizer = torch.optim.AdamW(alpha.parameters(), lr=lr)
print_params(alpha)

strassen = StrassenDemoDataset(max_len=max_len)
dl = DataLoader(strassen, batch_size=strassen.n_demos, shuffle=True)
for i_epoch in range(n_epochs):
    epoch_loss_pol = 0
    epoch_loss_val = 0
    for states, scalars, target_actions, rewards in dl:
        alpha.train()
        loss_pol, loss_val = alpha.fwd_train(states, scalars, target_actions, rewards)
        epoch_loss_pol += loss_pol
        epoch_loss_val += loss_val
        loss_combined = weight_pol * loss_pol + weigh_val * loss_val
        optimizer.zero_grad()
        loss_combined.backward()
        optimizer.step()
        if i_epoch % n_print == 0:
            print(
                f"epoch: {i_epoch} policy loss: {epoch_loss_pol} value loss {epoch_loss_val}"
            )
            alpha.eval()
            aa, pp, qq = alpha.fwd_infer(states, scalars)
            correct = torch.eq(aa.squeeze(), target_actions)
            print(f"Percent correct= {100*torch.mean(correct.float())}")
            print(f"Baseline= {100*torch.mean((target_actions == 2).float())}")
            pass


def play_strassen_game(alpha):
    strassen_tensor, action_list = get_strassen_tensor("cpu")
    strassen_input = strassen_tensor.unsqueeze(0).unsqueeze(0)
    current_state = strassen_input.clone()
    scalar_fixed = torch.tensor([[0.0]])
    print(f"start : {torch.sum(strassen_input != 0)} nonzero elements remaining")
    for ii in range(10):
        for st in strassen.state_tensor:
            if torch.all(torch.eq(st, current_state.squeeze(0))):
                print("Current state found in dataset")
                break
        aa, pp, qq = alpha.infer(current_state, scalar_fixed)
        # valid = True in [torch.all(torch.eq(aa.squeeze(), action_list[i])) for i in range(action_list.shape[0])]
        valid = False
        for j in range(action_list.shape[0]):
            if torch.all(torch.eq(aa.squeeze(), action_list[j])):
                valid = j
                break
        print(f"action: {aa.squeeze()} ; valid: {valid}")
        uu, vv, ww = torch.split(aa.squeeze() - 2, 4, dim=-1)
        action_tensor = factors_to_tensor((uu, vv, ww))
        current_state = current_state - action_tensor
        print(f"step {ii} : {torch.sum(current_state != 0)} nonzero elements remaining")
        if torch.sum(current_state != 0) == 0:
            break


def confirm_strassen_works():
    strassen_tensor, action_list = get_strassen_tensor("cpu")
    strassen_input = strassen_tensor.unsqueeze(0).unsqueeze(0)
    current_state = strassen_input.clone()
    print(f"start : {torch.sum(current_state != 0)} nonzero elements remaining")
    for ii in range(7):
        for st in strassen.state_tensor:
            if torch.all(torch.eq(st, current_state.squeeze(0))):
                print("Current state found in dataset")
                break
        ff = action_list[ii]
        uu, vv, ww = torch.split(ff - 2, 4, dim=-1)
        action_tensor = factors_to_tensor((uu, vv, ww))
        current_state = current_state - action_tensor
        print(f"step {ii} : {torch.sum(current_state != 0)} nonzero elements remaining")
