from time import time

from model import *
from synthetic_examples import *

torch.manual_seed(1337)
lr = 1e-3
max_iters = 2001
check_interval = 100
batch_size = 1
dim_3d = 4
dim_t = 1
dim_s = 1
dim_c = 5
n_logits = 3
n_samples = 32
n_steps = 12
n_quantile = 8
# alpha = AlphaTensor(
#     dim_3d=4,
#     dim_t=1,
#     dim_s=1,
#     dim_c=4,
#     n_samples=8,
#     n_steps=12,
#     n_logits=3,
#     n_feats=8,
#     n_heads=2,
# )
# optimizer = torch.optim.AdamW(alpha.parameters(), lr=lr)
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
# alpha = alpha.to(device)
def print_params(alpha):
    print(f"{sum(p.numel() for p in alpha.parameters()) // int(1e6)}M parameters")
    print(f"{sum(p.numel() for p in alpha.torso.parameters())} parameters: torso")
    print(f"{sum(p.numel() for p in alpha.policy_head.parameters()) // int(1e6)}M parameters: policy head")
    print(f"{sum(p.numel() for p in alpha.policy_head.parameters())} parameters: policy head")
    # print(f"{sum(p.numel() for p in alpha.value_head.parameters()) // int(1e3)}k parameters: value head")
    print(f"{sum(p.numel() for p in alpha.value_head.parameters())} parameters: value head")


def test_single_action():
    dim_t = 2
    xx = strassen_tensor.unsqueeze(0).unsqueeze(0)
    xx = torch.cat([xx for _ in range(dim_t)], 1)
    xx = torch.cat([xx for _ in range(batch_size)], 0)
    ss = torch.zeros(batch_size, dim_s)
    for i_action in range(strassen_steps.shape[0]):
        alpha = AlphaTensor(
            dim_3d=4,
            dim_t=dim_t,
            dim_s=1,
            dim_c=8,
            n_samples=4,
            n_steps=12,
            n_logits=3,
            n_feats=8,
            n_heads=2,
            n_hidden=8,
        )
        # print_params(alpha)
        optimizer = torch.optim.AdamW(alpha.parameters(), lr=lr)
        g_action = strassen_steps[i_action].unsqueeze(0)
        g_action = torch.cat([g_action for _ in range(batch_size)], 0)
        g_value = torch.ones(1).unsqueeze(0)
        g_value = torch.cat([g_value for _ in range(batch_size)], 0)
        for ii in range(max_iters):
            l_pol, l_val = alpha.train(xx, ss, g_action, g_value)
            ll = l_pol + l_val
            optimizer.zero_grad()
            ll.backward()
            optimizer.step()
            if ii % check_interval == 0:
                print(f"{ii} : policy: {l_pol} : value: {l_val}")
                print(f"target {i_action}: {g_action[0]}")
                aa, pp, qq = alpha.infer(xx[:1], ss[:1])
                print(f"output {i_action}: {aa[0, 0]}")
                print(f"{12 - torch.sum(torch.eq(aa[0,0], g_action[0]))} wrong elements")
                l_pol, l_val = alpha.train(xx, ss, g_action, g_value)
                a2, pp, qq = alpha.infer(xx[:1], ss[:1])
                if torch.equal(aa[0, 0], g_action[0]):
                    print("-------------------------------------------------------------------------MATCH!")
                    break

if __name__ == "__main__":
    test_single_action()
    # alpha = AlphaTensor(
    #     dim_3d=4,
    #     dim_t=1,
    #     dim_s=1,
    #     dim_c=4,
    #     n_samples=8,
    #     n_steps=12,
    #     n_logits=3,
    #     n_feats=8,
    #     n_heads=2,
    #     n_hidden=8,
    #     n_layers=2,
    # )
    # print_params(alpha)
    pass

    # t0 = time()
    # # aa, pp, qq = alpha.infer(xx, ss)
    #
    # g_action = strassen_steps[0].unsqueeze(0)
    # g_value = torch.ones(n_quantile).unsqueeze(0)
    # print(f"target: {g_action}")
    # for ii in range(max_iters):
    #     l_pol, l_val = alpha.train(xx, ss, g_action, g_value)
    #     ll = l_pol + l_val
    #     optimizer.zero_grad(set_to_none=True)
    #     ll.backward()
    #     optimizer.step()
    #     if ii % 10 == 0:
    #         print(f"{ii} : policy: {l_pol} : value: {l_val}")
    #         print(f"target: {g_action}")
    #         aa, pp, qq = alpha.infer(xx, ss)
    #         print(f"output: {aa[0,0]}")
    #         if torch.equal(aa[0, 0], g_action[0]):
    #             print("MATCH!")
    #
    # pass

    # for ii in range(max_iters):
    #     xx, ss, g_action, g_value = get_batch("train")
    #     l_pol, l_val = alpha.train(xx, ss, g_action, g_value)
    #     optimizer.zero_grad(set_to_none=True)
    #     l_pol.backward()
    #     l_val.backward()
    #     optimizer.step()
