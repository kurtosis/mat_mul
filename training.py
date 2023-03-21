from model import *
from synthetic_examples import *
from utils import *

torch.manual_seed(1337)
lr = 1e-3
max_iters = 2001
check_interval = 4
batch_size = 1
dim_3d = 4
dim_t = 1
dim_s = 1
dim_c = 5
n_logits = 3
n_samples = 32
n_steps = 12
n_quantile = 8


device = "cpu"

strassen_tensor, strassen_steps = get_strassen_tensor(device)


def test_single_action():
    dim_t = 2
    xx = strassen_tensor
    xx = xx.to(device)
    xx = xx.unsqueeze(0).unsqueeze(0)
    xx = torch.cat([xx for _ in range(dim_t)], 1)
    xx = torch.cat([xx for _ in range(batch_size)], 0)
    ss = torch.zeros(batch_size, dim_s, device=device)
    alpha = AlphaTensor(
        dim_3d=4,
        dim_t=dim_t,
        dim_s=1,
        dim_c=8,
        n_samples=1,
        n_steps=12,
        n_logits=4,
        n_feats=8,
        n_heads=2,
        n_hidden=8,
        device=device,
    )
    alpha = alpha.to(device)
    print_params(alpha)
    for i_action in range(strassen_steps.shape[0]):
        optimizer = torch.optim.AdamW(alpha.parameters(), lr=lr)
        g_action = strassen_steps[i_action].unsqueeze(0)
        # reserve 0 for <SOS> token (used in infer)
        g_action = g_action + 1
        g_action = torch.cat([g_action for _ in range(batch_size)], 0)
        g_value = torch.ones(1, device=device).unsqueeze(0)
        g_value = torch.cat([g_value for _ in range(batch_size)], 0)
        for ii in range(max_iters):
            alpha.train()
            l_pol, l_val = alpha.fwd_train(xx, ss, g_action, g_value)
            ll = l_pol + l_val
            optimizer.zero_grad()
            ll.backward()
            optimizer.step()
            if ii % check_interval == 0:
                print(f"{ii} : policy: {l_pol} : value: {l_val}")
                print(f"target {i_action}: {g_action[0]}")
                alpha.eval()
                aa, pp, qq = alpha.infer(xx[:1], ss[:1])
                print(f"output {i_action}: {aa[0, 0]}")
                print(
                    f"{12 - torch.sum(torch.eq(aa[0,0], g_action[0]))} wrong elements"
                )
                if torch.equal(aa[0, 0], g_action[0]):
                    print(
                        "-------------------------------------------------------------------------MATCH!"
                    )
                    break


if __name__ == "__main__":
    test_single_action()
