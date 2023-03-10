from model import *

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
