from math import ceil

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical

DROPOUT_PROB = 0.5


class Head(nn.Module):
    def __init__(self, c1, c2, d, causal_mask=False):
        super().__init__()
        self.d = d
        self.causal_mask = causal_mask
        self.query = nn.Linear(c1, d, bias=False)
        self.key = nn.Linear(c2, d, bias=False)
        self.value = nn.Linear(c2, d, bias=False)
        # TO DO: put this in buffer, pass nx, ny earlier
        # self.register_buffer("triu", torch.triu(torch.ones(nx, ny)))

    def forward(self, x, y):
        # x (*, nx, c1)
        # y (*, ny, c2)
        q = self.query(x)  # (*, nx, d)
        k = self.key(y)  # (*, ny, d)
        v = self.value(y)  # (*, ny, d)
        a = q @ k.transpose(-2, -1) * self.d**-0.5  # (*, nx, ny)
        # TO DO: is this the right softmax dim?
        a = F.softmax(a, dim=-1)  # (*, nx, ny)
        if self.causal_mask:
            nx = x.shape[-2]
            ny = y.shape[-2]
            triu = torch.triu(torch.ones(nx, ny))
            a = a.masked_fill(triu, 0)
        out = a @ v  # (*, nx, d)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, nx, ny, c1, c2, n_heads=16, d=32, w=4, **kwargs):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.c1 = c1
        self.c2 = c2
        self.n_heads = n_heads
        self.d = d
        self.w = w
        self.heads = nn.ModuleList(
            [Head(self.c1, self.c2, self.d, **kwargs) for _ in range(self.n_heads)]
        )
        self.li1 = nn.Linear(n_heads * self.d, self.c1)
        self.ln1 = nn.LayerNorm(self.c1)
        self.ln2 = nn.LayerNorm(self.c2)
        self.ln3 = nn.LayerNorm(self.c1)
        self.li2 = nn.Linear(self.c1, self.c1 * self.w)
        self.li3 = nn.Linear(self.c1 * self.w, self.c1)

    def forward(self, x, y):
        # x (*, nx, c1)
        # y (*, ny, c2)
        x = self.ln1(x)  # (*, nx, c1)
        y = self.ln2(y)  # (*, ny, c2)
        out = torch.cat([h(x, y) for h in self.heads], dim=-1)  # (*, nx, n_heads*d)
        out = x + self.li1(out)  # (*, nx, c1)
        out0 = out  # (*, nx, c1)
        out = self.ln3(out)  # (*, nx, c1)
        out = self.li2(out)  # (*, nx, c1*w)
        out = F.gelu(out)  # (*, nx, c1*w)
        out = self.li3(out)  # (*, nx, c1)
        out = out0 + out  # (*, nx, c1)
        return out


class AttentiveMode(nn.Module):
    # def __init__(self, nx, ny, c1, c2, dim_3d, **kwargs):
    def __init__(self, dim_3d, c1, **kwargs):
        super().__init__()
        # self.nx = nx
        # self.ny = ny
        self.c1 = c1
        # self.c2 = c2
        self.dim_3d = dim_3d
        # self.mha = MultiHeadAttention(self.nx, self.ny, self.c1, self.c2, **kwargs)
        self.mha = MultiHeadAttention(
            2 * self.dim_3d, 2 * self.dim_3d, self.c1, self.c1, **kwargs
        )

    def forward(self, g):
        # TO DO: make sure this can handle batching correctly
        # x1, x2, x3 = x_input
        # x1,x2,x3 are all (*, dim_3d,dim_3d,c)
        # TO DO: pass dims from init arguments
        # dim_3d = g[0].shape[-2]
        # g = [x1, x2, x3]
        for m1, m2 in [(0, 1), (1, 2), (2, 0)]:
            # TO DO: are these dims right?
            a = torch.cat(
                (g[m1], g[m2].transpose(-2, -3)), dim=-2
            )  # (*, dim_3d,2*dim_3d,c)
            # TO DO: make this parallel
            for i in range(self.dim_3d):
                cc = self.mha(a[i, :, :], a[i, :, :])  # (2*dim_3d, c)
                g[m1][i, :, :] = cc[: self.dim_3d, :]  # (dim_3d, c)
                g[m2][i, :, :] = cc[self.dim_3d :, :]  # (dim_3d, c)
                # Is below a bug in paper?
                # g[m2][i, :, :] = cc[dim_3d:, :].transpose(0, 1)
        return g  # [(*, dim_3d, dim_3d, c)]*3


class AttentiveModeBatch(nn.Module):
    def __init__(self, dim_3d, c1, **kwargs):
        super().__init__()
        self.c1 = c1
        self.dim_3d = dim_3d
        self.mha = MultiHeadAttention(
            2 * self.dim_3d, 2 * self.dim_3d, self.c1, self.c1, **kwargs
        )

    def forward(self, g):
        # x1, x2, x3 = x_input
        # x1,x2,x3 are all (*, dim_3d,dim_3d,c)
        for m1, m2 in [(0, 1), (1, 2), (2, 0)]:
            a = torch.cat(
                (g[m1], g[m2].transpose(-2, -3)), dim=-2
            )  # (*, dim_3d,2*dim_3d,c)
            # TO DO: make this parallel
            for i in range(self.dim_3d):
                cc = self.mha(a[:, i, :, :], a[:, i, :, :])  # (2*dim_3d, c)
                g[m1][:, i, :, :] = cc[:, : self.dim_3d, :]  # (dim_3d, c)
                g[m2][:, i, :, :] = cc[:, self.dim_3d :, :]  # (dim_3d, c)
                # Is below a bug in paper?
                # g[m2][i, :, :] = cc[dim_3d:, :].transpose(0, 1)
        return g  # [(*, dim_3d, dim_3d, c)]*3


class Torso(nn.Module):
    def __init__(self, dim_3d, dim_t, dim_s, dim_c, n_layers=8, **kwargs):
        super().__init__()
        self.dim_3d = dim_3d
        self.dim_t = dim_t
        self.dim_s = dim_s
        self.dim_c = dim_c
        self.ln1 = nn.LayerNorm(self.dim_c)
        self.ln2 = nn.LayerNorm(self.dim_c)
        self.li1 = nn.Linear(self.dim_s, self.dim_3d**2)
        self.li2 = nn.Linear(self.dim_3d * self.dim_t + 1, self.dim_c)
        self.blocks = nn.Sequential(
            *[
                AttentiveModeBatch(self.dim_3d, self.dim_c, **kwargs)
                for _ in range(n_layers)
            ]
            # * [AttentiveMode(self.dim_c, self.dim_c, **kwargs) for _ in range(n_layers)]
        )

    def forward(self, xx, ss):
        # assumes batch size dim!
        # xx (*, dim_t, dim_3d, dim_3d, dim_3d)
        # ss (*, dim_s)
        # x1 = xx.permute(1, 2, 3, 0).reshape(
        # x1 = xx.permute(-3, -2, -1, -4).reshape(
        x1 = xx.permute(0, 2, 3, 4, 1).reshape(
            -1, self.dim_3d, self.dim_3d, self.dim_3d * self.dim_t
        )  # (*, dim_3d, dim_3d, dim_3d*dim_t)
        # x2 = xx.permute(3, 1, 2, 0).reshape(
        # x2 = xx.permute(-1, -3, -2, -4).reshape(
        x2 = xx.permute(0, 4, 2, 3, 1).reshape(
            -1, self.dim_3d, self.dim_3d, self.dim_3d * self.dim_t
        )  # (*, dim_3d, dim_3d, dim_3d*dim_t)
        # x3 = xx.permute(2, 3, 1, 0).reshape(
        # x3 = xx.permute(-2, -1, -3, -4).reshape(
        x3 = xx.permute(0, 3, 4, 2, 1).reshape(
            -1, self.dim_3d, self.dim_3d, self.dim_3d * self.dim_t
        )  # (*, dim_3d, dim_3d, dim_3d*dim_t)
        g = [x1, x2, x3]  # [(*, dim_3d, dim_3d, dim_3d*dim_t)] * 3
        for i in range(3):
            p = self.li1(ss)  # (*, dim_3d**2)
            p = p.reshape(-1, self.dim_3d, self.dim_3d, 1)  # (*, dim_3d, dim_3d, 1)
            g[i] = torch.cat([g[i], p], dim=-1)  # (*, dim_3d, dim_3d, dim_3d*dim_t+1)
            g[i] = self.li2(g[i])  # (*, dim_3d, dim_3d, dim_c)
        # x1, x2, x3 = g
        # x1, x2, x3 = self.blocks(x1, x2, x3)
        g = self.blocks(g)  # [(*, dim_3d, dim_3d, dim_c)] * 3
        # ee = torch.stack(g, dim=1)  # (*, dim_3d, dim_3d, dim_c)
        ee = torch.stack(g, dim=-4)  # (*, 3, dim_3d, dim_3d, dim_c)
        # ee = torch.stack((x1, x2, x3), dim=1)
        ee = ee.reshape(-1, 3 * self.dim_3d**2, self.dim_c)  # (*, 3*dim_3d**2, dim_c)
        return ee


class PredictBlock(nn.Module):
    def __init__(self, n_steps, n_feats, n_heads, dim_m, dim_c):
        super().__init__()
        self.n_steps = n_steps
        self.n_feats = n_feats
        self.n_heads = n_heads
        self.dim_m = dim_m
        self.dim_c = dim_c
        self.ln1 = nn.LayerNorm(self.n_feats * self.n_heads)
        self.ln2 = nn.LayerNorm(self.n_feats * self.n_heads)
        self.dropout1 = nn.Dropout(DROPOUT_PROB)
        self.dropout2 = nn.Dropout(DROPOUT_PROB)
        self.att1 = MultiHeadAttention(
            self.n_steps,
            self.n_steps,
            self.n_feats * self.n_heads,
            self.n_feats * self.n_heads,
            n_heads=self.n_heads,
            causal_mask=True,
        )
        self.att2 = MultiHeadAttention(
            self.n_steps,
            self.dim_m,
            self.n_feats * self.n_heads,
            self.dim_c,
            n_heads=self.n_heads,
        )

    def forward(self, x_input, training=False):
        xx, ee = x_input  # xx (*, n_steps, n_feats*n_heads) ; ee (m, c)
        xx = self.ln1(xx)  # (*, n_steps, n_feats*n_heads)
        cc = self.att1(xx, xx)  # (*, n_steps, n_feats*n_heads)
        if training:
            cc = self.dropout1(cc)
        xx = xx + cc  # (*, n_steps, n_feats*n_heads)
        xx = self.ln2(xx)  # (*, n_steps, n_feats*n_heads)
        cc = self.att2(xx, ee)  # (*, n_steps, n_feats*n_heads)
        if training:
            cc = self.dropout2(cc)
        xx = xx + cc  # (*, n_steps, n_feats*n_heads)
        return xx, ee


class PredictActionLogits(nn.Module):
    def __init__(
        self, n_steps, n_logits, dim_m, dim_c, n_feats=64, n_heads=32, n_layers=2
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_logits = n_logits
        self.dim_m = dim_m
        self.dim_c = dim_c
        self.n_feats = n_feats
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.li1 = nn.Linear(self.n_logits, self.n_feats * self.n_heads)
        # TO DO: how many dims do we need pos enc over?
        self.pos_enc = nn.Parameter(
            torch.rand(self.n_steps, self.n_feats * self.n_heads)
        )
        self.blocks = nn.Sequential(
            *[
                PredictBlock(
                    self.n_steps, self.n_feats, self.n_heads, self.dim_m, self.dim_c
                )
                for _ in range(self.n_layers)
            ]
        )
        self.li2 = nn.Linear(self.n_feats * self.n_heads, self.n_logits)

    def forward(self, aa, ee):
        # aa (n_steps, n_logits) ; ee (dim_m, dim_c)
        xx = self.li1(aa)  # (n_steps, n_feats*n_heads)
        xx = xx + self.pos_enc  # (n_steps, n_feats*n_heads)
        xx, _ = self.blocks((xx, ee))  # (n_steps, n_feats*n_heads)
        oo = F.relu(xx)  # (n_steps, n_feats*n_heads)
        oo = self.li2(oo)  # (n_steps, n_logits)
        return oo, xx


class PolicyHead(nn.Module):
    def __init__(
        self, n_steps, n_logits, dim_m, dim_c, n_feats=64, n_heads=32, **kwargs
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_logits = n_logits
        self.dim_m = dim_m
        self.dim_c = dim_c
        self.n_feats = n_feats
        self.n_heads = n_heads
        self.predict_action_logits = PredictActionLogits(
            self.n_steps,
            self.n_logits,
            self.dim_m,
            self.dim_c,
            n_feats=self.n_feats,
            n_heads=self.n_heads,
            **kwargs,
        )

    def train(self, ee, gg):
        # ee (*, dim_m, dim_c) ; gg (*, n_steps)
        gg = gg.type(torch.LongTensor).roll(1)  # (n_steps)
        gg = F.one_hot(gg, num_classes=self.n_logits).type(
            torch.FloatTensor
        )  # (*, n_steps, n_logits)
        oo, zz = self.predict_action_logits(
            gg, ee
        )  # oo (*, n_steps, n_logits) ; zz (*, n_steps, n_feats*n_heads)
        return oo, zz[:, 0, :]

    def infer(self, ee, n_samples=32):
        batch_size = ee.shape[0]
        aa = torch.zeros(batch_size, n_samples, self.n_steps, dtype=torch.long)
        pp = torch.ones(batch_size, n_samples)
        oo = torch.zeros(batch_size, n_samples, self.n_steps, self.n_logits)
        zz = torch.zeros(batch_size, self.n_steps, self.n_feats * self.n_heads)
        for s in range(n_samples):
            for i in range(self.n_steps):
                gg = F.one_hot(aa[:, s, :], num_classes=self.n_logits).type(
                    torch.FloatTensor
                )  # (batch_size, n_samples, n_steps, n_logits)
                oo[:, s, :, :], zz = self.predict_action_logits(gg, ee)
                distrib = Categorical(logits=oo[:, s, i, :])
                aa[:, s, i] = distrib.sample()
                p_i = distrib.probs[
                    torch.arange(batch_size), aa[:, s, i]
                ]  # (batch_size)
                pp[:, s] = torch.mul(pp[:, s], p_i)
        return (
            aa,
            pp,
            zz[:, 0, :],
        )  # (b, n_samples, n_steps), (b, n_samples), (b, n_feats*n_heads)

    def infer_broadcast(self, ee, n_samples=32):
        """Don't use this right now.
        TO DO: fix this."""
        batch_size = ee.shape[0]
        aa = torch.zeros(batch_size, n_samples, self.n_steps, dtype=torch.long)
        pp = torch.ones(batch_size, n_samples)
        oo = torch.zeros(batch_size, n_samples, self.n_steps, self.n_logits)
        zz = torch.zeros(batch_size, self.n_steps, self.n_feats * self.n_heads)
        aa = aa.view(batch_size * n_samples, self.n_steps)
        pp = pp.view(batch_size * n_samples)
        oo = oo.view(batch_size * n_samples, self.n_steps, self.n_logits)
        # for s in range(n_samples):
        for i in range(self.n_steps):
            gg = F.one_hot(aa, num_classes=self.n_logits).type(
                torch.FloatTensor
            )  # (batch_size*n_samples, n_steps, n_logits)
            oo, zz = self.predict_action_logits(
                gg, ee
            )  # oo (batch_size*n_samples, n_steps, n_logits) ; zz (batch_size, n_steps, n_feats*n_heads)
            distrib = Categorical(logits=oo[:, i, :])
            aa[:, i] = distrib.sample()  # (), (n_logits)
            p_i = distrib.probs[torch.arange(batch_size), aa[:, i]]
            pp = torch.mul(pp, p_i)
        return (
            aa,
            pp,
            zz[:, 0, :],
        )  # (n_samples, n_steps), (n_samples), (n_feats*n_heads)


class ValueHead(nn.Module):
    def __init__(self, dim_c, n_out=8):
        super().__init__()
        self.dim_c = dim_c
        self.n_out = n_out
        self.mlp = nn.Sequential(
            nn.Linear(self.dim_c, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.n_out),
        )
        # self.li1 = nn.Linear(self.dim_c, 512)
        # self.li2 = nn.Linear(512, 512)
        # self.li3 = nn.Linear(512, 512)
        # self.li4 = nn.Linear(512, self.n)

    def forward(self, xx):
        # xx (dim_c)
        # xx = F.relu(self.li1(xx))  # (512)
        # xx = F.relu(self.li2(xx))  # (512)
        # xx = F.relu(self.li3(xx))  # (512)
        qq = self.mlp(xx)  # (n_out)
        return qq


def quantile_loss(qq, gg, n, delta=1):
    # qq (n) ; gg (n)
    tau = (torch.arange(n, dtype=torch.float32) + 0.5) / n  # (n)
    dd = gg - qq  # (n)
    hh = F.huber_loss(gg, qq, reduction="none", delta=delta)  # (n)
    kk = torch.abs(tau - (dd < 0).float())  # (n)
    return torch.mean(torch.mul(hh, kk))  # ()


class AlphaTensor(nn.Module):
    def __init__(self, dim_3d, dim_t, dim_s, dim_c, n_samples, n_steps, n_logits):
        super().__init__()
        self.dim_3d = dim_3d
        self.dim_t = dim_t
        self.dim_s = dim_s
        self.dim_c = dim_c
        self.n_samples = n_samples
        self.n_steps = n_steps
        self.n_logits = n_logits
        self.torso = Torso(self.dim_3d, self.dim_t, self.dim_s, self.dim_c)
        self.policy_head = PolicyHead(
            self.n_steps, self.n_logits, 3 * self.dim_3d**2, self.dim_c
        )
        # TO DO: figure out how to run 2048 dim through
        self.value_head = ValueHead(2048)

    # def quantile_loss(self, qq, aa, rr):
    #     pass

    def value_risk_mgmt(self, qq, uq=0.75):
        # qq (batch_size, n)
        jj = ceil(uq * qq.shape[-1]) - 1
        return qq[:, jj:].mean()

    def train(self, xx, ss, g_action, g_value):
        ee = self.torso(xx, ss)  # (3*dim_3d**2, dim_c)
        oo, zz = self.policy_head.train(
            ee, g_action, self.n_logits
        )  # oo (n_steps, n_logits) ; zz (n_feats*n_heads)
        l_pol = torch.sum(F.cross_entropy(oo, g_action))
        # TO DO: the dims don't seem correct here
        qq = self.value_head(zz)  # (n)
        l_val = quantile_loss(qq, g_value)
        return l_pol, l_val

    def infer(self, xx, ss):
        ee = self.torso(xx, ss)  # (3*dim_3d**2, dim_c)
        aa, pp, zz = self.policy_head.infer(
            ee, self.n_samples
        )  # aa (n_samples, n_steps) ; pp (n_samples) ; zz (n_feats*n_heads)
        qq = self.value_head(zz)  # (n)
        qq = self.value_risk_mgmt(qq)  # ()
        return aa, pp, qq


if __name__ == "__main__":
    pass
