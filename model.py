from math import ceil

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical


def create_fixed_positional_encoding(n_position: int, n_embedding: int, device: str):
    pe = torch.zeros(n_position, n_embedding, device=device)
    positions = torch.arange(n_position)
    denominators = 10000 ** (-torch.arange(0, n_embedding, 2) / n_embedding)
    pe[:, 0::2] = torch.outer(positions, denominators).sin()
    pe[:, 1::2] = torch.outer(positions, denominators).cos()
    return pe


class Head(nn.Module):
    def __init__(self, c1: int, c2: int, d: int, causal_mask=False, **kwargs):
        super().__init__()
        self.d = d
        self.causal_mask = causal_mask
        self.query = nn.Linear(c1, d, bias=False)
        self.key = nn.Linear(c2, d, bias=False)
        self.value = nn.Linear(c2, d, bias=False)
        # TO DO: put this in buffer, pass nx, ny earlier
        # self.register_buffer("triu", torch.triu(torch.ones(nx, ny), dim=1))

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # x (*, nx, c1)
        # y (*, ny, c2)
        q = self.query(x)  # (*, nx, d)
        k = self.key(y)  # (*, ny, d)
        v = self.value(y)  # (*, ny, d)
        a = q @ k.transpose(-2, -1) / (self.d ** 0.5)  # (*, nx, ny)
        if self.causal_mask:
            b = torch.tril(torch.ones_like(a))
            a = a.masked_fill(b == 0, float("-inf"))
        a = F.softmax(a, dim=-1)  # (*, nx, ny)
        out = a @ v  # (*, nx, d)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, c1: int, c2: int, n_heads=16, d=32, w=4, **kwargs):
        super().__init__()
        self.ln1 = nn.LayerNorm(c1)
        self.ln2 = nn.LayerNorm(c2)
        self.heads = nn.ModuleList([Head(c1, c2, d, **kwargs) for _ in range(n_heads)])
        self.li1 = nn.Linear(n_heads * d, c1)
        self.ln3 = nn.LayerNorm(c1)
        self.li2 = nn.Linear(c1, c1 * w)
        self.li3 = nn.Linear(c1 * w, c1)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # x (*, nx, c1)
        # y (*, ny, c2)
        x_norm = self.ln1(x)  # (*, nx, c1)
        y_norm = self.ln2(y)  # (*, ny, c2)
        x_out = torch.cat(
            [h(x_norm, y_norm) for h in self.heads], dim=-1
        )  # (*, nx, n_heads*d)
        x_out = x + self.li1(x_out)  # (*, nx, c1)
        out = self.ln3(x_out)  # (*, nx, c1)
        out = self.li2(out)  # (*, nx, c1*w)
        out = F.gelu(out)  # (*, nx, c1*w)
        out = self.li3(out)  # (*, nx, c1)
        return x_out + out


class AttentiveModeBatch(nn.Module):
    def __init__(self, dim_3d: int, c1: int, **kwargs):
        super().__init__()
        self.dim_3d = dim_3d
        self.mha = MultiHeadAttention(c1, c1, **kwargs)

    def forward(self, g: list[torch.Tensor]):
        # x1, x2, x3 = x_input
        # x1,x2,x3 are all (*, dim_3d,dim_3d,c)
        for m1, m2 in [(0, 1), (1, 2), (2, 0)]:
            # TO DO: confirm transpose is correct
            a = torch.cat((g[m1], g[m2]), dim=-2)  # (*, dim_3d,2*dim_3d,c)
            # a = torch.cat(
            #     (g[m1], g[m2].transpose(-2, -3)), dim=-2
            # )  # (*, dim_3d,2*dim_3d,c)
            # TO DO: make this parallel
            cc = self.mha(a, a)
            g[m1] = cc[:, :, : self.dim_3d, :]
            g[m2] = cc[:, :, self.dim_3d :, :]
            # for i in range(self.dim_3d):
            #     cc = self.mha(a[:, i, :, :], a[:, i, :, :])  # (2*dim_3d, c)
            #     g[m1][:, i, :, :] = cc[:, : self.dim_3d, :]  # (dim_3d, c)
            #     g[m2][:, i, :, :] = cc[:, self.dim_3d :, :]  # (dim_3d, c)
            # Is below a bug in paper?
            # g[m2][i, :, :] = cc[dim_3d:, :].transpose(0, 1)
        return g  # [(*, dim_3d, dim_3d, c)]*3


class Torso(nn.Module):
    def __init__(
        self, dim_3d: int, dim_t: int, dim_s: int, dim_c: int, n_layers=8, **kwargs
    ):
        super().__init__()
        self.dim_3d = dim_3d
        self.dim_t = dim_t
        self.dim_c = dim_c
        self.li1 = nn.ModuleList([nn.Linear(dim_s, dim_3d ** 2) for _ in range(3)])
        self.li2 = nn.ModuleList(
            [nn.Linear(dim_3d * dim_t + 1, dim_c) for _ in range(3)]
        )
        self.blocks = nn.Sequential(
            *[AttentiveModeBatch(dim_3d, dim_c, **kwargs) for _ in range(n_layers)]
        )

    def forward(self, xx: torch.Tensor, ss: torch.Tensor):
        # xx (*, dim_t, dim_3d, dim_3d, dim_3d)
        # ss (*, dim_s)
        x1 = xx.permute(0, 2, 3, 4, 1).reshape(
            -1, self.dim_3d, self.dim_3d, self.dim_3d * self.dim_t
        )  # (*, dim_3d, dim_3d, dim_3d*dim_t)
        x2 = xx.permute(0, 4, 2, 3, 1).reshape(
            -1, self.dim_3d, self.dim_3d, self.dim_3d * self.dim_t
        )  # (*, dim_3d, dim_3d, dim_3d*dim_t)
        x3 = xx.permute(0, 3, 4, 2, 1).reshape(
            -1, self.dim_3d, self.dim_3d, self.dim_3d * self.dim_t
        )  # (*, dim_3d, dim_3d, dim_3d*dim_t)
        g = [x1, x2, x3]  # [(*, dim_3d, dim_3d, dim_3d*dim_t)] * 3
        for i in range(3):
            p = self.li1[i](ss)  # (*, dim_3d**2)
            p = p.reshape(-1, self.dim_3d, self.dim_3d, 1)  # (*, dim_3d, dim_3d, 1)
            g[i] = torch.cat([g[i], p], dim=-1)  # (*, dim_3d, dim_3d, dim_3d*dim_t+1)
            g[i] = self.li2[i](g[i])  # (*, dim_3d, dim_3d, dim_c)
        g = self.blocks(g)  # [(*, dim_3d, dim_3d, dim_c)] * 3
        ee = torch.stack(g, dim=2)  # (*, 3, dim_3d, dim_3d, dim_c)
        ee = ee.reshape(-1, 3 * self.dim_3d ** 2, self.dim_c)  # (*, 3*dim_3d**2, dim_c)
        return ee


# Algorithm A.4.a
class PredictBlock(nn.Module):
    def __init__(self, n_feats: int, n_heads: int, dim_c: int, drop_p=0.5, **kwargs):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_feats * n_heads)
        self.att1 = MultiHeadAttention(
            n_feats * n_heads, n_feats * n_heads, n_heads=n_heads, causal_mask=True,
        )
        self.dropout1 = nn.Dropout(drop_p)
        self.ln2 = nn.LayerNorm(n_feats * n_heads)
        self.att2 = MultiHeadAttention(n_feats * n_heads, dim_c, n_heads=n_heads,)
        self.dropout2 = nn.Dropout(drop_p)

    def forward(self, xx: torch.Tensor, ee: torch.Tensor):
        xx = self.ln1(xx)  # (*, n_steps, n_feats*n_heads)
        # Self attention
        cc = self.att1(xx, xx)  # (*, n_steps, n_feats*n_heads)
        cc = self.dropout1(cc)
        xx = xx + cc  # (*, n_steps, n_feats*n_heads)
        xx = self.ln2(xx)  # (*, n_steps, n_feats*n_heads)
        # Cross attention
        cc = self.att2(xx, ee)  # (*, n_steps, n_feats*n_heads)
        cc = self.dropout2(cc)
        xx = xx + cc  # (*, n_steps, n_feats*n_heads)
        return xx


# Algorithm A.4
class PredictActionLogits(nn.Module):
    def __init__(
        self,
        n_steps: int,
        n_logits: int,
        dim_c: int,
        n_feats=64,
        n_heads=32,
        n_layers=4,
        device="cpu",
        **kwargs
    ):
        super().__init__()
        self.emb1 = nn.Embedding(n_logits, n_feats * n_heads)
        self.pos_enc = nn.Parameter(torch.rand(n_steps, n_feats * n_heads))
        pos_enc_fix = create_fixed_positional_encoding(
            n_steps, n_feats * n_heads, device
        )
        self.register_buffer("pos_enc_fix", pos_enc_fix)
        self.blocks = nn.Sequential(
            *[PredictBlock(n_feats, n_heads, dim_c, **kwargs) for _ in range(n_layers)]
        )
        self.li1 = nn.Linear(n_feats * n_heads, n_logits)

    def forward(self, aa: torch.Tensor, ee: torch.Tensor, **kwargs):
        # aa (n_steps, n_logits) ; ee (dim_m, dim_c)
        xx = self.emb1(aa)  # (n_steps, n_feats*n_heads)
        xx = (
            xx + self.pos_enc[: xx.shape[1]] + self.pos_enc_fix[: xx.shape[1]]
        )  # (n_steps, n_feats*n_heads)
        for block in self.blocks:
            xx = block(xx, ee)
        oo = F.relu(xx)  # (n_steps, n_feats*n_heads)
        oo = self.li1(oo)  # (n_steps, n_logits)
        return oo, xx


class PolicyHead(nn.Module):
    def __init__(self, n_steps: int, n_logits: int, dim_c: int, device="cpu", **kwargs):
        super().__init__()
        self.n_steps = n_steps
        self.device = device
        self.predict_action_logits = PredictActionLogits(
            n_steps, n_logits, dim_c, **kwargs,
        )

    def fwd_train(self, ee: torch.Tensor, gg: torch.Tensor):
        # ee (B, dim_m, dim_c) ; gg (B, n_steps)
        if self.device == "mps":
            gg_shifted = torch.zeros_like(gg, dtype=torch.long)
            gg_shifted[:, 1:] = gg[:, :-1]
            gg = gg_shifted
        else:
            gg = gg.long().roll(shifts=1, dims=1)  # (n_steps)
            gg[:, 0] = 0

        oo, zz = self.predict_action_logits(
            gg, ee
        )  # oo (*, n_steps, n_logits) ; zz (*, n_steps, n_feats*n_heads)
        return oo, zz[:, 0, :]

    def fwd_infer(self, ee: torch.Tensor, n_samples=32):
        batch_size = ee.shape[0]
        aa = torch.zeros(
            batch_size,
            n_samples,
            self.n_steps + 1,
            dtype=torch.long,
            device=self.device,
        )
        pp = torch.ones(batch_size, n_samples, device=self.device)
        # TO DO: understand these lines
        ee = ee.unsqueeze(1).repeat(1, n_samples, 1, 1)  # (1, n_samples, dim_m, dim_c)
        aa = aa.view(-1, self.n_steps + 1)  # (1*n_samples, n_steps)
        pp = pp.view(-1)  # (1*n_samples)
        ee = ee.view(-1, ee.shape[-2], ee.shape[-1])  # (1*n_samples, dim_m, dim_c)
        for i in range(self.n_steps):
            oo_s, zz_s = self.predict_action_logits(aa[:, : i + 1], ee)
            distrib = Categorical(logits=oo_s[:, i])
            aa[:, i + 1] = distrib.sample()  # allow to sample 0, but reserve for <SOS>
            p_i = distrib.probs[torch.arange(batch_size), aa[:, i + 1]]  # (batch_size)
            pp = torch.mul(pp, p_i)
        return (
            aa[:, 1:].view(batch_size, n_samples, self.n_steps),
            pp.view(batch_size, n_samples),
            zz_s[:, 0].view(batch_size, n_samples, *zz_s.shape[2:]).mean(1),
        )  # (b, n_samples, n_steps), (b, n_samples), (b, n_feats*n_heads)


class ValueHead(nn.Module):
    def __init__(self, n_feats=64, n_heads=32, n_hidden=512, n_quantile=8, **kwargs):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_feats * n_heads, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_quantile),
        )

    def forward(self, xx: torch.Tensor):
        return self.mlp(xx)  # (n_quantile)


def quantile_loss(qq: torch.Tensor, gg: torch.Tensor, delta=1, device="cpu"):
    # qq (n) ; gg (*)
    n = qq.shape[-1]
    # TO DO: store tau in buffer?
    tau = (torch.arange(n, dtype=torch.float32, device=device) + 0.5) / n  # (n)
    hh = F.huber_loss(gg, qq, reduction="none", delta=delta)  # (n)
    dd = gg - qq  # (n)
    # TO DO: is the sign of dd correct?
    kk = torch.abs(tau - (dd > 0).float())  # (n)
    # flipped sign from paper
    # kk = torch.abs(tau - (dd < 0).float())  # (n)
    return torch.mean(torch.mul(hh, kk))  # ()


class AlphaTensor(nn.Module):
    def __init__(
        self,
        dim_3d=4,
        dim_t=8,
        dim_s=1,
        dim_c=16,
        n_samples=32,
        n_steps=12,
        n_logits=3,
        device="cpu",
        **kwargs
    ):
        super().__init__()
        # self.dim_3d = dim_3d
        # self.dim_t = dim_t
        # self.dim_s = dim_s
        # self.dim_c = dim_c
        self.n_samples = n_samples
        # self.n_steps = n_steps
        self.n_logits = n_logits
        self.device = device
        self.torso = Torso(dim_3d, dim_t, dim_s, dim_c, **kwargs)
        self.policy_head = PolicyHead(n_steps, n_logits, dim_c, device=device, **kwargs)
        # TO DO: figure out how to run 2048 dim through
        self.value_head = ValueHead(**kwargs)

    @staticmethod
    def value_risk_mgmt(qq: torch.Tensor, uq=0.75):
        # qq (batch_size, n)
        # TO DO: can make this int?
        jj = ceil(uq * qq.shape[-1]) - 1
        return torch.mean(qq[:, jj:], dim=-1)

    def fwd_train(
        self,
        xx: torch.Tensor,
        ss: torch.Tensor,
        g_action: torch.Tensor,
        g_value: torch.Tensor,
    ):
        ee = self.torso(xx, ss)  # (3*dim_3d**2, dim_c)
        oo, zz = self.policy_head.fwd_train(
            ee, g_action
        )  # oo (*, n_steps, n_logits) ; zz (*, n_feats*n_heads)
        l_pol = F.cross_entropy(
            oo.view(-1, self.n_logits), g_action.view(-1), reduction="sum"
        )
        # TO DO: the dims don't seem correct here
        qq = self.value_head(zz)  # (n)
        l_val = quantile_loss(qq, g_value, device=self.device)
        return l_pol, l_val

    def fwd_infer(self, xx: torch.Tensor, ss: torch.Tensor):
        ee = self.torso(xx, ss)  # (3*dim_3d**2, dim_c)
        aa, pp, z1 = self.policy_head.fwd_infer(
            ee, self.n_samples
        )  # aa (*, n_samples, n_steps) ; pp (*, n_samples) ; z1 (*, n_feats*n_heads)
        qq = self.value_head(z1)  # (n)
        qq = self.value_risk_mgmt(qq)  # (1)
        return aa, pp, qq


if __name__ == "__main__":
    pass
