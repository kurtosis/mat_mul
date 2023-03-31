from argparse import ArgumentParser
import datetime
import hashlib
import json
import logging

# import logging.handlers
import os
import sys

# import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from data_generation import *
from model import *

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class TrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]
        parser = ArgumentParser()
        parser.add_argument("--lr", type=float, default=5e-4)
        parser.add_argument("--dropout_p", type=float, default=0.5)
        parser.add_argument("--max_iters", type=int, default=10)
        parser.add_argument("--max_len", type=int, default=None)
        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--dim_3d", type=int, default=4)
        parser.add_argument("--dim_t", type=int, default=2)
        parser.add_argument("--dim_s", type=int, default=1)
        parser.add_argument("--dim_c", type=int, default=8)
        parser.add_argument("--n_samples", type=int, default=1)
        parser.add_argument("--n_steps", type=int, default=12)
        parser.add_argument("--n_logits", type=int, default=4)
        parser.add_argument("--n_feats", type=int, default=8)
        parser.add_argument("--n_heads", type=int, default=4)
        parser.add_argument("--n_hidden", type=int, default=8)
        parser.add_argument("--device", type=str, default="cpu")
        parser.add_argument("--n_demos", type=int, default=100)
        parser.add_argument("--n_epochs", type=int, default=100)
        parser.add_argument("--n_print", type=int, default=1)
        parser.add_argument("--n_act", type=int, default=1)
        parser.add_argument("--weight_pol", type=int, default=1)
        parser.add_argument("--weigh_val", type=int, default=0)
        parser.add_argument("--tb_prefix", type=str, default="synth_demo")
        parser.add_argument(
            "comment",
            type=str,
            help="Comment suffix for Tensorboard run",
            nargs="?",
            default="synth",
        )
        parser.add_argument(
            "model_file",
            type=str,
            help="File name to load model params from.",
            nargs="?",
            default=None,
            # default="synth_2023-03-29_10.31.07_synth_1170.pt"
        )
        self.args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
        self.training_samples_count = 0
        self.trn_writer = None
        self.val_writer = None
        self.model = self.init_model()
        self.optimizer = self.init_optimizer()

    def init_model(self):
        model = AlphaTensor(
            self.args.dim_3d,
            self.args.dim_t,
            self.args.dim_s,
            self.args.dim_c,
            self.args.n_steps,
            self.args.n_logits,
            n_feats=self.args.n_feats,
            n_heads=self.args.n_heads,
            n_hidden=self.args.n_hidden,
            dropout_p=self.args.dropout_p,
            device=self.args.device,
        )
        model.to(self.args.device)
        return model

    def init_optimizer(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)

    def init_dl(self):
        pass

    def init_tensorboard_writers(self):
        if self.trn_writer is None:
            log_dir = Path("runs").joinpath(self.args.tb_prefix)
            self.trn_writer = SummaryWriter(
                log_dir=log_dir.joinpath(self.time_str + "-trn-" + self.args.comment)
            )
            self.val_writer = SummaryWriter(
                log_dir=log_dir.joinpath(self.time_str + "-val-" + self.args.comment)
                # log_dir=log_dir + "-val-" + self.args.comment
            )

    def log_metrics(self, *args, **kwargs):
        pass

    def save_model(self, type_str, i_epoch):
        file_path = Path("data_unversioned").joinpath(
            "models",
            self.args.tb_prefix,
            f"{type_str}_{self.time_str}_{self.args.comment}_{self.training_samples_count}.pt",
        )
        Path.mkdir(file_path.parent, mode=0o755, exist_ok=True)
        model = self.model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        state = {
            "model_name": type(model).__name__,
            # "optimizer_state": self.optimizer.state_dict(),
            "optimizer_name": type(self.optimizer).__name__,
            "epoch": i_epoch,
            "training_samples_count": self.training_samples_count,
            "dim_3d": model.dim_3d,
            "device": model.device,
            "n_logits": model.n_logits,
        }
        torch.save(model.state_dict(), file_path)
        # save parameters in a json file
        state_file_path = Path("data_unversioned").joinpath(
            "models",
            self.args.tb_prefix,
            f"{type_str}_{self.time_str}_{self.args.comment}_{self.training_samples_count}.json",
        )
        with open(state_file_path, "w") as f:
            json.dump(state, f)
        log.debug(f"Saved model params to {file_path}")
        with open(file_path, "rb") as f:
            log.info(f"SHA!: {hashlib.sha1(f.read()).hexdigest()}")

    def load_model(self, model_file):
        file_path = Path("data_unversioned").joinpath(
            "models",
            self.args.tb_prefix,
            model_file,
        )
        # d = torch.load(self.cli_args.finetune, map_location="cpu")
        self.model.load_state_dict(torch.load(file_path))
        self.model.eval()

    def main(self):
        pass


class SyntheticDemoTrainingApp(TrainingApp):
    def __init__(self):
        super().__init__()

    def init_dl(self):
        demos = SyntheticDemoDataset(
            self.args.dim_t,
            self.args.n_demos,
            self.args.dim_t,
            self.args.dim_3d,
            self.args.device,
        )
        demos_train, demos_test = random_split(demos, [0.9, 0.1])
        dl_train = DataLoader(
            demos_train, batch_size=self.args.batch_size, shuffle=True
        )
        dl_test = DataLoader(demos_test, batch_size=self.args.batch_size, shuffle=True)
        return dl_train, dl_test

    def log_metrics(self, i_epoch, mode_str, epoch_loss_pol, epoch_loss_val):
        self.init_tensorboard_writers()
        log.info(f"E{i_epoch} {type(self).__name__}")
        writer = getattr(self, mode_str + "_writer")
        writer.add_scalar("loss_policy", epoch_loss_pol, self.training_samples_count)
        writer.add_scalar("loss_value", epoch_loss_val, self.training_samples_count)

    def _take_action(self, states, scalars):
        aa, pp, qq = self.model.fwd_infer(
            states, scalars, n_samples=self.args.n_samples
        )
        aa[aa == 0] = 2  # hack to avoid choosing <SOS> token
        uu, vv, ww = torch.split(aa.squeeze() - 2, 4, dim=-1)
        action_tensor = factors_to_tensor((uu, vv, ww))
        new_states = states - action_tensor
        rank_ubs = torch.sum(new_states != 0, [-1, -2, -3], dtype=torch.int32)
        best_samples = torch.min(rank_ubs, -1)
        return new_states, scalars + 1, best_samples

    def main(self):
        if self.args.model_file is not None:
            print("loading model")
            self.load_model(self.args.model_file)
        else:
            print("NOT loading model")
        dl_train, dl_test = self.init_dl()
        for i_epoch in range(self.args.n_epochs):
            epoch_loss_pol = 0
            epoch_loss_val = 0
            # training epoch
            self.model.train()
            for states, scalars, target_actions, rewards in dl_train:
                loss_pol, loss_val = self.model.fwd_train(
                    states, scalars, target_actions, rewards
                )
                epoch_loss_pol += loss_pol
                epoch_loss_val += loss_val
                loss_combined = (
                    self.args.weight_pol * loss_pol + self.args.weigh_val * loss_val
                )
                self.optimizer.zero_grad()
                loss_combined.backward()
                self.optimizer.step()
            self.training_samples_count += len(dl_train.dataset)
            epoch_loss_pol /= len(dl_train.dataset)
            epoch_loss_val /= len(dl_train.dataset)
            self.log_metrics(i_epoch, "trn", epoch_loss_pol, epoch_loss_val)
            # train/validation loss printout
            if i_epoch % self.args.n_print == 0:
                print(
                    f"TRN epoch: {i_epoch} policy loss: {epoch_loss_pol} "
                    f"value loss {epoch_loss_val}"
                )
                self.model.eval()
                epoch_loss_pol = 0
                epoch_loss_val = 0
                for states, scalars, target_actions, rewards in dl_test:
                    loss_pol, loss_val = self.model.fwd_train(
                        states, scalars, target_actions, rewards
                    )
                    epoch_loss_pol += loss_pol
                    epoch_loss_val += loss_val

                epoch_loss_pol /= len(dl_test.dataset)
                epoch_loss_val /= len(dl_test.dataset)
                self.log_metrics(i_epoch, "val", epoch_loss_pol, epoch_loss_val)
                self.save_model("synth", i_epoch)
                print(
                    f"VAL epoch: {i_epoch} policy loss: {epoch_loss_pol} "
                    f"value loss {epoch_loss_val}"
                )
            # Solution search printout
            if i_epoch + 1 % self.args.n_act == 0:
                self.model.eval()
                lowest_rank = torch.tensor(self.model.dim_3d**3)
                num_solutions_found = 0
                for states, scalars, _, _ in dl_test:
                    _, _, best_samples = self._take_action(states, scalars)
                    lowest_rank = torch.min(lowest_rank, torch.min(best_samples.values))
                    num_solutions_found += torch.sum(best_samples.values == 0)
                if num_solutions_found > 0:
                    print(
                        f"E{i_epoch}: Found {num_solutions_found} solutions out of {len(dl_test.dataset)}"
                    )
                else:
                    print(f"E{i_epoch} : lowest rank found = {lowest_rank}")


if __name__ == "__main__":
    SyntheticDemoTrainingApp().main()
