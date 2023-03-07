import torch
from torch import nn
from torch.nn import functional as F


class Head(nn.Module):
    def __init__(self, c1, c2, d, causal_mask=False):
        super().__init__()
        self.d = d
        self.causal_mask = causal_mask
        self.query = nn.Linear(c1, self.d, bias=False)
        self.key = nn.Linear(c2, self.d, bias=False)
        self.value = nn.Linear(c2, self.d, bias=False)
        # self.register_buffer("triu", torch.triu(torch.ones(nx, ny)))

    def forward(self, x, y):
        # _, nx, _ = x.shape
        # _, ny, _ = y.shape
        nx, _ = x.shape
        ny, _ = y.shape
        q = self.query(x)  # (b, nx, d)
        k = self.key(y)  # (b, ny, d)
        v = self.value(y)  # (b, ny, d)
        a = q @ k.transpose(-2, -1) * self.d**-0.5  # (b, nx, ny)
        if self.causal_mask:
            triu = torch.triu(torch.ones(nx, ny))
            a = a.masked_fill(triu, 0)
        out = a @ v  # (b, nx, d)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, c1, c2, num_heads=16, d=32, w=4, **kwargs):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(c1, c2, d, **kwargs) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(num_heads * d, c1)
        self.ln1 = nn.LayerNorm(c1)
        self.li1 = nn.Linear(c1, c1 * w)
        self.li2 = nn.Linear(c1 * w, c1)

    def forward(self, x, y):
        out = torch.cat([h(x, y) for h in self.heads], dim=-1)  # (b, nx, num_heads*d)
        out = x + self.proj(out)  # (b, nx, c1)
        out0 = out
        out = self.ln1(out)
        out = self.li1(out)
        out = F.gelu(out)
        out = self.li2(out)
        out = out0 + out
        return out


class Block(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mha = MultiHeadAttention(*args, **kwargs)

    def forward(self, g):
        # x1, x2, x3 = x_input
        # x1,x2,x3 are all (S,S,c)
        dim_3d = g[0].shape[1]
        # g = [x1, x2, x3]
        for m1, m2 in [(0, 1), (1, 2), (2, 0)]:
            a = torch.cat((g[m1], g[m2].transpose(0, 1)), dim=1)  # (S,2S,c)
            for i in range(dim_3d):
                cc = self.mha(a[i, :, :], a[i, :, :])  # (2S,c)
                g[m1][i, :, :] = cc[:dim_3d, :]
                g[m2][i, :, :] = cc[dim_3d:, :]
                # Is below a bug in paper?
                # g[m2][i, :, :] = cc[dim_3d:, :].transpose(0, 1)
        return g


class Torso(nn.Module):
    def __init__(self, dim_3d, dim_t, dim_s, dim_c, n_layer=8, **kwargs):
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
            *[Block(self.dim_c, self.dim_c, **kwargs) for _ in range(n_layer)]
        )

    def forward(self, xx, ss):
        # xx (dim_t,dim_3d,dim_3d,dim_3d)
        # ss (dim_s)
        x1 = xx.permute(1, 2, 3, 0).reshape(
            self.dim_3d, self.dim_3d, self.dim_3d * self.dim_t
        )
        x2 = xx.permute(3, 1, 2, 0).reshape(
            self.dim_3d, self.dim_3d, self.dim_3d * self.dim_t
        )
        x3 = xx.permute(2, 3, 1, 0).reshape(
            self.dim_3d, self.dim_3d, self.dim_3d * self.dim_t
        )
        g = [x1, x2, x3]
        for i in range(3):
            p = self.li1(ss)
            p = p.reshape(self.dim_3d, self.dim_3d, 1)  # (dim_3d, dim_3d, 1)
            g[i] = torch.cat([g[i], p], dim=-1)  # (dim_3d, dim_3d, dim_3d*dim_t+1)
            g[i] = self.li2(g[i])  # (dim_3d, dim_3d, dim_c)
        # x1, x2, x3 = g
        # x1, x2, x3 = self.blocks(x1, x2, x3)
        g = self.blocks(g)
        ee = torch.stack(g, dim=1)
        # ee = torch.stack((x1, x2, x3), dim=1)
        ee = ee.reshape(3 * self.dim_3d**2, self.dim_c)
        return ee


if __name__ == "__main__":
    batch_size = 16
    dim_3d0 = 2
    dim_t0 = 3
    dim_s0 = 4
    dim_c0 = 5
    torso1 = Torso(2, 3, 4, 5)
    xx0 = torch.rand(dim_t0, dim_3d0, dim_3d0, dim_3d0)
    ss0 = torch.rand(dim_s0)
    ff = torso1(xx0, ss0)
    print(ff.shape)

