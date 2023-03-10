from model import *

# for testing
batch_size = 16
dim_3d = 2
dim_t = 3
dim_s = 4
dim_c = 5
dim_c2 = 7
dim_d = 3
nx = 11
ny = 13
n_steps = 5
n_feats = 16
n_heads = 32
dim_m = 17
n_logits = 3
n_samples = 4

def test_head():
    print("Test Head")
    # check that Head works on batch
    head0 = Head(dim_c, dim_c2, dim_d)
    x0 = torch.rand(nx, dim_c)
    y0 = torch.rand(ny, dim_c2)
    out = head0(x0, y0)
    print(out.shape)
    assert out.shape == (nx, dim_d)
    print("Batch")
    x0 = torch.rand(batch_size, nx, dim_c)
    y0 = torch.rand(batch_size, ny, dim_c2)
    out = head0(x0, y0)
    print(out.shape)
    assert out.shape == (batch_size, nx, dim_d)


def test_mha():
    print("MHA")
    mha0 = MultiHeadAttention(nx, ny, dim_c, dim_c2)
    x0 = torch.rand(nx, dim_c)
    y0 = torch.rand(ny, dim_c2)
    ff = mha0(x0, y0)
    print(f"out {ff.shape}")
    assert ff.shape == (nx, dim_c)
    print("Batch")
    x0 = torch.rand(batch_size, nx, dim_c)
    y0 = torch.rand(batch_size, ny, dim_c2)
    ff = mha0(x0, y0)
    print(f"out {ff.shape}")
    assert ff.shape == (batch_size, nx, dim_c)


def test_attentive_mode():
    print("AttentiveMode")
    attm = AttentiveModeBatch(dim_3d, dim_c)
    # xx1 = torch.rand(dim_3d, dim_3d, dim_c)
    # xx2 = torch.rand(dim_3d, dim_3d, dim_c)
    # xx3 = torch.rand(dim_3d, dim_3d, dim_c)
    # g = [xx1, xx2, xx3]
    # ff = attm(g)
    # print(f"out len {len(ff)}")
    # print(f"out shape {ff[0].shape}")
    # assert len(ff) == 3
    # assert ff[0].shape == xx1.shape
    print("Batch")
    xx1 = torch.rand(batch_size, dim_3d, dim_3d, dim_c)
    xx2 = torch.rand(batch_size, dim_3d, dim_3d, dim_c)
    xx3 = torch.rand(batch_size, dim_3d, dim_3d, dim_c)
    g = [xx1, xx2, xx3]
    ff = attm(g)
    print(f"out len {len(ff)}")
    print(f"out shape {ff[0].shape}")
    assert len(ff) == 3
    assert ff[0].shape == xx1.shape


def test_torso():
    print("Torso")
    torso = Torso(dim_3d, dim_t, dim_s, dim_c)
    # xx = torch.rand(dim_t, dim_3d, dim_3d, dim_3d)
    # ss = torch.rand(dim_s)
    # ff = torso(xx, ss)
    # print(f"out {ff.shape}")
    # assert ff.shape == (3*dim_3d**2, dim_c)
    xx = torch.rand(batch_size, dim_t, dim_3d, dim_3d, dim_3d)
    ss = torch.rand(batch_size, dim_s)
    ee = torso(xx, ss)
    print(f"out {ee.shape}")
    assert ee.shape == (batch_size, 3 * dim_3d**2, dim_c)


def test_pred_block():
    print("Test PredictBlock")
    pb0 = PredictBlock(n_steps, n_feats, n_heads, dim_m, dim_c)
    x0 = torch.rand(n_steps, n_feats * n_heads)
    e0 = torch.rand(dim_m, dim_c)
    out = pb0((x0, e0))
    print(f"x out {out[0].shape}")
    print(f"e out {out[1].shape}")
    assert out[0].shape == x0.shape
    assert out[1].shape == e0.shape
    print("Batch")
    x0 = torch.rand(batch_size, n_steps, n_feats * n_heads)
    e0 = torch.rand(batch_size, dim_m, dim_c)
    pb0 = PredictBlock(n_steps, n_feats, n_heads, dim_m, dim_c)
    out = pb0((x0, e0))
    print(f"x out {out[0].shape}")
    print(f"e out {out[1].shape}")
    assert out[0].shape == x0.shape
    assert out[1].shape == e0.shape


def test_pred_action_logits():
    print("Test PredictActionLogits")
    pal0 = PredictActionLogits(
        n_steps, n_logits, dim_m, dim_c, n_feats=n_feats, n_heads=n_heads
    )
    a0 = torch.rand(n_steps, n_logits)
    e0 = torch.rand(dim_m, dim_c)
    out = pal0(a0, e0)
    print(f"o out {out[0].shape}")
    print(f"x out {out[1].shape}")
    assert out[0].shape == (n_steps, n_logits)
    assert out[1].shape == (n_steps, n_feats * n_heads)
    print("Batch")
    a0 = torch.rand(batch_size, n_steps, n_logits)
    e0 = torch.rand(batch_size, dim_m, dim_c)
    out = pal0(a0, e0)
    print(f"o out {out[0].shape}")
    print(f"x out {out[1].shape}")
    assert out[0].shape == (batch_size, n_steps, n_logits)
    assert out[1].shape == (batch_size, n_steps, n_feats * n_heads)


def test_policy_head(
    n_steps=4, n_logits=3, gg=torch.tensor([[1, 0, 2, 2]]), batch_size=1
):
    pi = PolicyHead(n_steps, n_logits, dim_m, dim_c)
    torso = Torso(dim_3d, dim_t, dim_s, dim_c)
    xx = torch.rand(batch_size, dim_t, dim_3d, dim_3d, dim_3d)
    ss = torch.rand(batch_size, dim_s)
    ee = torso(xx, ss)
    oo, z1 = pi.train(ee, gg)
    print(f"train oo {oo.shape}")
    print(f"train z1 {z1.shape}")
    assert oo.shape == (batch_size, n_steps, n_logits)
    assert z1.shape == (
        batch_size,
        pi.predict_action_logits.n_feats * pi.predict_action_logits.n_heads,
    )
    # aa, pp, z1 = pi.infer_broadcast(ee, n_samples=32)
    aa, pp, z1 = pi.infer(ee, n_samples=32)
    assert aa.shape == (batch_size, 32, pi.predict_action_logits.n_steps)
    assert pp.shape == (batch_size, 32)
    assert z1.shape == (
        batch_size,
        pi.predict_action_logits.n_feats * pi.predict_action_logits.n_heads,
    )


def test_value_head():
    vv = ValueHead(dim_c)
    xx = torch.rand(dim_c)
    qq = vv(xx)
    assert qq.shape == (vv.n_out,)


def test_quantile_loss(n = 8, batch_size=16):
    qq = torch.rand(n)
    gg = torch.rand(n)
    ll = quantile_loss(qq, gg, n)
    assert ll.shape == torch.Size([])
    # for batch, I think we still want to reduce loss over batch?
    qq = torch.rand(batch_size, n)
    gg = torch.rand(batch_size, n)
    ll = quantile_loss(qq, gg, n)
    assert ll.shape == torch.Size([])

def test_alpha_tensor(batch_size=2):
    alpha = AlphaTensor(dim_3d, dim_t, dim_s, dim_c, n_samples, n_steps, n_logits)
    xx = torch.rand(batch_size, dim_t, dim_3d, dim_3d, dim_3d)
    ss = torch.rand(batch_size, dim_s)
    aa, pp, qq = alpha.infer(xx, ss)
    assert aa.shape == (batch_size, n_samples, n_steps)
    assert pp.shape == (batch_size, n_samples)
    assert qq == torch.Size([])



if __name__ == "__main__":
    # test_head()
    # test_pred_block()
    # test_pred_action_logits()
    # test_mha()
    # test_attentive_mode()
    # test_torso()
    # test_policy_head()
    # test_value_head()
    # test_quantile_loss()
    test_alpha_tensor()
