from argparse import ArgumentParser
from pathlib import Path
from time import time

from model import *
from data_generation import *
from strassen_training import *
from synthetic_training import *
from utils import *


def main():
    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout_p", type=float, default=0.0)
    parser.add_argument("--max_iters", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--dim_3d", type=int, default=4)
    parser.add_argument("--dim_t", type=int, default=1)
    parser.add_argument("--dim_s", type=int, default=1)
    parser.add_argument("--dim_c", type=int, default=5)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--n_steps", type=int, default=12)
    parser.add_argument("--n_logits", type=int, default=4)
    parser.add_argument("--n_feats", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_hidden", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--task", type=str, default="synthetic")
    args = parser.parse_args()

    alpha = AlphaTensor(
        args.dim_3d,
        args.dim_t,
        args.dim_s,
        args.dim_c,
        args.n_samples,
        args.n_steps,
        args.n_logits,
        n_feats=args.n_feats,
        n_heads=args.n_heads,
        n_hidden=args.n_hidden,
        dropout_p=args.dropout_p,
        device=args.device,
    )
    alpha.to(args.device)
    optimizer = torch.optim.AdamW(alpha.parameters(), lr=args.lr)

    print_params(alpha)

    if args.task == "strassen":
        train_strassen(alpha, optimizer)
    elif args.task == "synthetic":
        train_synthetic(alpha, optimizer)


if __name__ == "__main__":
    main()
