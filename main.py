from time import time

from model import *

if __name__ == "__main__":
    lr = 1e-4
    max_iters = 10
    batch_size = 16
    dim_3d = 4
    dim_t = 7
    dim_s = 1
    dim_c = 5
    n_logits = 3
    n_samples = 32
    n_steps = 10
    alpha = AlphaTensor(dim_3d, dim_t, dim_s, dim_c, n_samples, n_steps, n_logits)
    # aa = alpha.to(device)
    print(f"{sum(p.numel() for p in alpha.parameters())//int(1e6)}M parameters")
    optimizer = torch.optim.AdamW(alpha.parameters(), lr=lr)

    t0 = time()
    for ii in range(max_iters):
        xx, ss, g_action, g_value = get_batch("train")
        l_pol, l_val = alpha.train(xx, ss, g_action, g_value)
        optimizer.zero_grad(set_to_none=True)
        l_pol.backward()
        l_val.backward()
        optimizer.step()

distrib = Categorical(logits=oo)
distrib.sample()
F.cross_entropy(oo.view(-1, self.n_logits), g_action.view(-1), reduction='sum')


# g_action = torch.tensor([2, 1, 1, 1, 1, 2, 1, 0, 1, 2, 1, 2])
g_test = g_action.type(torch.LongTensor).roll(1)  # (n_steps)
g_test = F.one_hot(g_test, num_classes=self.n_logits).type(
    torch.FloatTensor
)  # (*, n_steps, n_logits)

xx, yy = self.predict_action_logits(g_test, ee)

g2 = gg
g2[0,5] = torch.tensor([1., 0., 0.])
g2[0,6] = torch.tensor([1., 0., 0.])
x2, y2 = self.predict_action_logits(g2, ee)
