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
    assert out.shape == (nx, dim_d)
    print(" Singleton passed")
    x0 = torch.rand(batch_size, nx, dim_c)
    y0 = torch.rand(batch_size, ny, dim_c2)
    out = head0(x0, y0)
    assert out.shape == (batch_size, nx, dim_d)
    print(" Batch passed")


def test_mha():
    print("Test MultiHeadAttention")
    mha0 = MultiHeadAttention(nx, ny, dim_c, dim_c2)
    x0 = torch.rand(nx, dim_c)
    y0 = torch.rand(ny, dim_c2)
    out = mha0(x0, y0)
    assert out.shape == (nx, dim_c)
    print(" Singleton passed")
    x0 = torch.rand(batch_size, nx, dim_c)
    y0 = torch.rand(batch_size, ny, dim_c2)
    out = mha0(x0, y0)
    assert out.shape == (batch_size, nx, dim_c)
    print(" Batch passed")


def test_attentive_mode():
    print("Test AttentiveMode")
    attm = AttentiveModeBatch(dim_3d, dim_c)
    xx1 = torch.rand(batch_size, dim_3d, dim_3d, dim_c)
    xx2 = torch.rand(batch_size, dim_3d, dim_3d, dim_c)
    xx3 = torch.rand(batch_size, dim_3d, dim_3d, dim_c)
    g = [xx1, xx2, xx3]
    out = attm(g)
    assert len(out) == 3
    assert out[0].shape == xx1.shape
    print(" Batch passed")


def test_torso():
    print("Test Torso")
    torso = Torso(dim_3d, dim_t, dim_s, dim_c)
    xx = torch.rand(batch_size, dim_t, dim_3d, dim_3d, dim_3d)
    ss = torch.rand(batch_size, dim_s)
    ee = torso(xx, ss)
    assert ee.shape == (batch_size, 3 * dim_3d**2, dim_c)
    print(" Batch passed")


def test_pred_block():
    print("Test PredictBlock")
    pb0 = PredictBlock(n_steps, n_feats, n_heads, dim_m, dim_c)
    x0 = torch.rand(n_steps, n_feats * n_heads)
    e0 = torch.rand(dim_m, dim_c)
    out = pb0((x0, e0))
    assert out[0].shape == x0.shape
    assert out[1].shape == e0.shape
    print(" Singleton passed")
    x0 = torch.rand(batch_size, n_steps, n_feats * n_heads)
    e0 = torch.rand(batch_size, dim_m, dim_c)
    out = pb0((x0, e0))
    assert out[0].shape == x0.shape
    assert out[1].shape == e0.shape
    print(" Batch passed")


def test_pred_action_logits():
    print("Test PredictActionLogits")
    pal0 = PredictActionLogits(
        n_steps, n_logits, dim_m, dim_c, n_feats=n_feats, n_heads=n_heads
    )
    a0 = torch.rand(n_steps, n_logits)
    e0 = torch.rand(dim_m, dim_c)
    out = pal0(a0, e0)
    assert out[0].shape == (n_steps, n_logits)
    assert out[1].shape == (n_steps, n_feats * n_heads)
    print(" Singleton passed")
    a0 = torch.rand(batch_size, n_steps, n_logits)
    e0 = torch.rand(batch_size, dim_m, dim_c)
    out = pal0(a0, e0)
    assert out[0].shape == (batch_size, n_steps, n_logits)
    assert out[1].shape == (batch_size, n_steps, n_feats * n_heads)
    print(" Batch passed")


def test_policy_head(
    n_steps=4, n_logits=3, gg=torch.tensor([[1, 0, 2, 2]]), batch_size=1
):
    print("Test PolicyHead")
    pi = PolicyHead(n_steps, n_logits, dim_m, dim_c)
    torso = Torso(dim_3d, dim_t, dim_s, dim_c)
    xx = torch.rand(batch_size, dim_t, dim_3d, dim_3d, dim_3d)
    ss = torch.rand(batch_size, dim_s)
    ee = torso(xx, ss)
    oo, z1 = pi.train(ee, gg)
    assert oo.shape == (batch_size, n_steps, n_logits)
    assert z1.shape == (
        batch_size,
        pi.predict_action_logits.n_feats * pi.predict_action_logits.n_heads,
    )
    print(" Train passed")
    aa, pp, z1 = pi.infer(ee, n_samples=32)
    assert aa.shape == (batch_size, 32, pi.predict_action_logits.n_steps)
    assert pp.shape == (batch_size, 32)
    assert z1.shape == (
        batch_size,
        pi.predict_action_logits.n_feats * pi.predict_action_logits.n_heads,
    )
    print(" Infer passed")


def test_value_head():
    print("Test ValueHead")
    vv = ValueHead(dim_c)
    xx = torch.rand(dim_c)
    qq = vv(xx)
    assert qq.shape == (vv.n_out,)
    print(" Singleton passed")


def test_quantile_loss(n=8, batch_size=16):
    print("Test quantile_loss")
    qq = torch.rand(n)
    gg = torch.rand(n)
    ll = quantile_loss(qq, gg)
    assert ll.shape == torch.Size([])
    print(" Singleton passed")
    qq = torch.rand(batch_size, n)
    gg = torch.rand(batch_size, n)
    ll = quantile_loss(qq, gg)
    assert ll.shape == torch.Size([])
    print(" Batch passed")


def test_alpha_tensor(batch_size=2):
    print("Test AlphaTensor")
    alpha = AlphaTensor(dim_3d, dim_t, dim_s, dim_c, n_samples, n_steps, n_logits)
    xx = torch.rand(batch_size, dim_t, dim_3d, dim_3d, dim_3d)
    ss = torch.rand(batch_size, dim_s)
    aa, pp, qq = alpha.infer(xx, ss)
    assert aa.shape == (batch_size, n_samples, n_steps)
    assert pp.shape == (batch_size, n_samples)
    assert qq.shape == torch.Size([])
    print(" Infer passed")
    g_action = torch.randint(
        n_logits,
        (
            batch_size,
            n_steps,
        ),
    )
    g_value = torch.rand((batch_size, 8))
    l_pol, l_val = alpha.train(xx, ss, g_action, g_value)
    assert l_pol.shape == l_val.shape == torch.Size([])
    print(" Train passed")


if __name__ == "__main__":
    test_head()
    test_pred_block()
    test_pred_action_logits()
    test_mha()
    test_attentive_mode()
    test_torso()
    test_policy_head()
    test_value_head()
    test_quantile_loss()
    test_alpha_tensor()
