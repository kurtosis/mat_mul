from math import ceil, sqrt

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical

DROPOUT_PROB = 0.5


def create_fixed_positional_encoding(n_position, n_embedding):
    pe = torch.zeros(n_position, n_embedding)
    positions = torch.arange(n_position)  # .unsqueeze(1)
    denominators = 10000 ** (-torch.arange(0, n_embedding, 2) / n_embedding)
    pe[:, 0::2] = torch.outer(positions, denominators).sin()
    pe[:, 1::2] = torch.outer(positions, denominators).cos()
    return pe

# class PositionalEncoding(nn.Module):
#     def __init__(self, n_position, n_embedding):
#         super().__init__()
#         pe = torch.zeros(n_position, n_embedding)
#         positions = torch.arange(n_position)  # .unsqueeze(1)
#         denominators = 10000 ** (-torch.arange(0, n_embedding, 2) / n_embedding)
#         pe[:, 0::2] = torch.outer(positions, denominators).sin()
#         pe[:, 1::2] = torch.outer(positions, denominators).cos()
#         self.register_buffer("pe", pe)


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

    def forward(self, x, y):
        # x (*, nx, c1)
        # y (*, ny, c2)
        q = self.query(x)  # (*, nx, d)
        k = self.key(y)  # (*, ny, d)
        v = self.value(y)  # (*, ny, d)
        a = q @ k.transpose(-2, -1) * self.d**-0.5  # (*, nx, ny)
        a = F.softmax(a, dim=-1)  # (*, nx, ny)
        if self.causal_mask:
            a = torch.triu(a, diagonal=1)
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
        self.heads = nn.ModuleList([Head(c1, c2, d, **kwargs) for _ in range(n_heads)])
        self.li1 = nn.Linear(n_heads * d, c1)
        self.ln1 = nn.LayerNorm(c1)
        self.ln2 = nn.LayerNorm(c2)
        self.ln3 = nn.LayerNorm(c1)
        self.li2 = nn.Linear(c1, c1 * w)
        self.li3 = nn.Linear(c1 * w, c1)

    def forward(self, x, y):
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


class AttentiveMode(nn.Module):
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
    def __init__(self, dim_3d, dim_t, dim_s, dim_c, n_layers=8, **kwargs):
        super().__init__()
        self.dim_3d = dim_3d
        self.dim_t = dim_t
        self.dim_s = dim_s
        self.dim_c = dim_c
        self.n_layers = n_layers
        self.ln1 = nn.LayerNorm(self.dim_c)
        self.ln2 = nn.LayerNorm(self.dim_c)
        self.li1 = nn.ModuleList(
            [nn.Linear(self.dim_s, self.dim_3d**2) for _ in range(3)]
        )
        self.li2 = nn.ModuleList(
            [nn.Linear(self.dim_3d * self.dim_t + 1, self.dim_c) for _ in range(3)]
        )
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
        ee = ee.reshape(-1, 3 * self.dim_3d**2, self.dim_c)  # (*, 3*dim_3d**2, dim_c)
        return ee


# Algorithm A.4.a
class PredictBlock(nn.Module):
    def __init__(self, n_steps, n_feats, n_heads, dim_m, dim_c, **kwargs):
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
        self.lin_ee = nn.Linear(
            self.dim_m * self.dim_c, self.n_steps * self.n_feats * self.n_heads
        )

    def forward(self, xx, ee, training):
        xx = self.ln1(xx)  # (*, n_steps, n_feats*n_heads)
        # Self attention
        cc = self.att1(xx, xx)  # (*, n_steps, n_feats*n_heads)
        if training:
            cc = self.dropout1(cc)
        xx = xx + cc  # (*, n_steps, n_feats*n_heads)
        xx = self.ln2(xx)  # (*, n_steps, n_feats*n_heads)
        # Cross attention
        cc = self.att2(xx, ee)  # (*, n_steps, n_feats*n_heads)
        if training:
            cc = self.dropout2(cc)
        xx = xx + cc  # (*, n_steps, n_feats*n_heads)
        # xx = cc
        return xx

        # # Linear layer replacement
        # cc = self.lin_ee(ee.view(ee.shape[0], -1))
        # cc = cc.view(xx.shape)
        # return cc


# Algorithm A.4
class PredictActionLogits(nn.Module):
    def __init__(
        self,
        n_steps,
        n_logits,
        dim_m,
        dim_c,
        n_feats=64,
        n_heads=32,
        n_layers=2,
        **kwargs
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_logits = n_logits
        self.dim_m = dim_m
        self.dim_c = dim_c
        self.n_feats = n_feats
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.emb1 = nn.Embedding(self.n_logits, self.n_feats * self.n_heads)
        # TO DO: figure out if we need to init this to small values for faster convergence
        # self.emb1.weight.data.normal_(0, 0.01)
        self.li1 = nn.Linear(self.n_logits, self.n_feats * self.n_heads)
        # TO DO: how many dims do we need pos enc over?
        # self.pos_enc = nn.Embedding(
        #     self.n_steps, self.n_feats * self.n_heads
        # )  # (*,1) -> (*,C) (n_embed)
        self.pos_enc = nn.Parameter(
            torch.rand(self.n_steps, self.n_feats * self.n_heads)
        )
        pos_enc_fix = create_fixed_positional_encoding(self.n_steps, self.n_feats * self.n_heads)
        self.register_buffer("pos_enc_fix", pos_enc_fix)
        self.blocks = nn.Sequential(
            *[
                PredictBlock(
                    self.n_steps, self.n_feats, self.n_heads, self.dim_m, self.dim_c
                )
                for _ in range(self.n_layers)
            ]
        )
        self.li2 = nn.Linear(self.n_feats * self.n_heads, self.n_logits)

    def forward(self, aa, ee, training=False, **kwargs):
        # aa (n_steps, n_logits) ; ee (dim_m, dim_c)
        xx = self.emb1(aa)  # (n_steps, n_feats*n_heads)
        # if training:
        #     dr = nn.Dropout()
        #     xx = dr(xx)
        # xx = xx + self.pos_enc[:xx.shape[1]]  # (n_steps, n_feats*n_heads)
        # xx = self.pos_enc[:xx.shape[1]]  # (n_steps, n_feats*n_heads)
        xx = self.pos_enc[:xx.shape[1]] + self.pos_enc_fix[:xx.shape[1]]  # (n_steps, n_feats*n_heads)
        for block in self.blocks:
            xx = block(xx, ee, training)
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
            n_feats=n_feats,
            n_heads=n_heads,
            **kwargs,
        )

    def train(self, ee: torch.Tensor, gg: torch.Tensor):
        # ee (B, dim_m, dim_c) ; gg (B, n_steps)
        gg = gg.type(torch.LongTensor).roll(shifts=-1, dims=1)  # (n_steps)
        # TO DO: make sure we can pass this through without one_hot
        # gg = F.one_hot(gg, num_classes=self.n_logits).type(
        #     torch.FloatTensor
        # )  # (*, n_steps, n_logits)
        oo, zz = self.predict_action_logits(
            gg,
            ee,
            training=True,
        )  # oo (*, n_steps, n_logits) ; zz (*, n_steps, n_feats*n_heads)
        return oo, zz[:, 0, :]

    def infer(self, ee: torch.Tensor, n_samples=32):
        batch_size = ee.shape[0]
        aa = torch.zeros(batch_size, n_samples, self.n_steps, dtype=torch.long)
        pp = torch.ones(batch_size, n_samples)
        # TO DO: understand these lines
        ee = ee.unsqueeze(1).repeat(1, n_samples, 1, 1)  # (1, n_samples, dim_m, dim_c)
        aa = aa.view(-1, self.n_steps)  # (1*n_samples, n_steps)
        pp = pp.view(-1)  # (1*n_samples)
        ee = ee.view(-1, ee.shape[-2], ee.shape[-1])  # (1*n_samples, dim_m, dim_c)
        # oo = torch.zeros(batch_size, self.n_steps, self.n_logits)
        # zz = torch.zeros(batch_size, self.n_steps, self.n_feats * self.n_heads)
        for i in range(self.n_steps):
            # gg = F.one_hot(aa[:, i], num_classes=self.n_logits).type(
            #     torch.FloatTensor
            # )  # (batch_size, n_samples, n_steps, n_logits)
            oo_s, zz_s = self.predict_action_logits(aa[:, :i + 1], ee)
            distrib = Categorical(logits=oo_s[:, i])
            aa[:, i] = distrib.sample()
            p_i = distrib.probs[torch.arange(batch_size), aa[:, i]]  # (batch_size)
            pp = torch.mul(pp, p_i)
        return (
            aa.view(batch_size, n_samples, self.n_steps),
            pp.view(batch_size, n_samples),
            zz_s[:, 0].view(batch_size, n_samples, *zz_s.shape[2:]).mean(1),
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

    def forward(self, xx):
        return self.mlp(xx)  # (n_quantile)


def quantile_loss(qq, gg, delta=1):
    # qq (n) ; gg (*)
    n = qq.shape[-1]
    # TO DO: store tau in buffer?
    tau = (torch.arange(n, dtype=torch.float32) + 0.5) / n  # (n)
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
        **kwargs
    ):
        super().__init__()
        self.dim_3d = dim_3d
        self.dim_t = dim_t
        self.dim_s = dim_s
        self.dim_c = dim_c
        self.n_samples = n_samples
        self.n_steps = n_steps
        self.n_logits = n_logits
        self.torso = Torso(dim_3d, dim_t, dim_s, dim_c, **kwargs)
        self.policy_head = PolicyHead(
            n_steps, n_logits, 3 * dim_3d**2, dim_c, **kwargs
        )
        # TO DO: figure out how to run 2048 dim through
        self.value_head = ValueHead(**kwargs)

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {
                nn.Linear,
                nn.Conv3d,
                nn.Conv2d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
            }:
                nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(
                        m.weight.data
                    )
                    bound = 1 / sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def value_risk_mgmt(self, qq, uq=0.75):
        # qq (batch_size, n)
        # TO DO: can make this int?
        jj = ceil(uq * qq.shape[-1]) - 1
        return torch.mean(qq[:, jj:], dim=-1)

    def train(self, xx, ss, g_action, g_value):
        ee = self.torso(xx, ss)  # (3*dim_3d**2, dim_c)
        oo, zz = self.policy_head.train(
            ee, g_action
        )  # oo (*, n_steps, n_logits) ; zz (*, n_feats*n_heads)
        l_pol = F.cross_entropy(
            oo.view(-1, self.n_logits), g_action.view(-1), reduction="sum"
        )
        # TO DO: the dims don't seem correct here
        qq = self.value_head(zz)  # (n)
        l_val = quantile_loss(qq, g_value)
        return l_pol, l_val

    def infer(self, xx, ss):
        ee = self.torso(xx, ss)  # (3*dim_3d**2, dim_c)
        aa, pp, z1 = self.policy_head.infer(
            ee, self.n_samples
        )  # aa (*, n_samples, n_steps) ; pp (*, n_samples) ; z1 (*, n_feats*n_heads)
        qq = self.value_head(z1)  # (n)
        qq = self.value_risk_mgmt(qq)  # (1)
        return aa, pp, qq


if __name__ == "__main__":
    pass
