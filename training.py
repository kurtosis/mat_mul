from argparse import ArgumentParser
import datetime
import hashlib
import json
import logging
import sys
import time

from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from act import *
from datasets import *
from model import *


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
log.addHandler(stream_handler)


class TrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]
        parser = ArgumentParser()
        parser.add_argument("--lr_initial", type=float, default=1e-3)
        parser.add_argument("--lr_final", type=float, default=1e-4)
        parser.add_argument("--lr_decay_epochs", type=int, default=10)
        parser.add_argument("--len_data", type=int, default=20000)
        parser.add_argument("--n_epochs", type=int, default=201)

        # MC parameters
        parser.add_argument(
            "--n_act",
            help="Frequency (in epochs) of MC tree search",
            type=int,
            default=1,
        )
        parser.add_argument(
            "--n_games",
            help="Number of games to play in MC act stage",
            type=int,
            default=16,
        )
        parser.add_argument(
            "--max_actions",
            help="Maximum number of actions to take in an MC trajectory",
            type=int,
            default=4,
        )
        parser.add_argument(
            "--n_sim",
            help="Number of simulations to run at each step of MC tree search",
            type=int,
            default=4,
        )
        parser.add_argument(
            "--n_samples",
            help="Number of actions to sample at each step in MC tree search",
            type=int,
            default=8,
        )

        parser.add_argument(
            "--n_val", help="Frequency (in epochs) of validation", type=int, default=10
        )
        parser.add_argument(
            "--n_save", help="Frequency (in epochs) of model save", type=int, default=10
        )
        parser.add_argument("--batch_size", type=int, default=256)

        # Model dimensionality parameters
        parser.add_argument("--dim_t", type=int, default=2)
        parser.add_argument("--dim_s", type=int, default=1)
        parser.add_argument("--dim_c", type=int, default=8)
        parser.add_argument("--n_feats", type=int, default=8)
        parser.add_argument("--n_heads", type=int, default=4)
        parser.add_argument("--n_hidden", type=int, default=128)

        # Matrix multiplication parameters
        parser.add_argument("--dim_3d", type=int, default=4)
        parser.add_argument(
            "--n_steps",
            help="Number of steps in a complete action",
            type=int,
            default=12,
        )
        parser.add_argument(
            "--n_logits", help="Cardinality of action token set", type=int, default=3
        )
        parser.add_argument("--device", type=str, default="cpu")
        parser.add_argument("--weight_pol", type=int, default=1)
        parser.add_argument("--weight_val", type=int, default=1000)
        parser.add_argument(
            "--n_bar",
            type=int,
            default=100,
            help="N_bar parameter for policy improvement temperature.",
        )
        parser.add_argument("--tb_prefix", type=str, default="tensor_game")
        parser.add_argument("--fract_synth", type=float, default=0.90)
        parser.add_argument("--fract_best", type=float, default=0.0)
        parser.add_argument("--start_rank", type=int, default=1)
        parser.add_argument("--dropout_p", type=float, default=0.5)
        parser.add_argument(
            "--model_file",
            type=str,
            help="File name to load model params from.",
            default=None,
        )
        parser.add_argument(
            "comment",
            type=str,
            help="Comment suffix for Tensorboard run",
            nargs="?",
            default="tg",
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
            self.args.n_samples,
            n_feats=self.args.n_feats,
            n_heads=self.args.n_heads,
            n_hidden=self.args.n_hidden,
            dropout_p=self.args.dropout_p,
            device=self.args.device,
        )
        model.to(self.args.device)
        return model

    def init_optimizer(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.args.lr_initial)

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
            )

    def log_metrics(self, i_epoch, mode_str, epoch_loss_pol, epoch_loss_val):
        self.init_tensorboard_writers()
        log.info(f"E{i_epoch} {self.training_samples_count} {type(self).__name__}")
        log.info(f"E{i_epoch} {mode_str} loss_policy {epoch_loss_pol}")
        log.info(f"E{i_epoch} {mode_str} loss_value  {epoch_loss_val}")
        writer = getattr(self, mode_str + "_writer")
        writer.add_scalar("loss_policy", epoch_loss_pol, self.training_samples_count)
        writer.add_scalar("loss_value", epoch_loss_val, self.training_samples_count)

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
            "optimizer_name": type(self.optimizer).__name__,
            "epoch": i_epoch,
            "training_samples_count": self.training_samples_count,
            "args": vars(self.args),
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
        log.debug(f"Saved model params to {state_file_path}")
        with open(file_path, "rb") as f:
            log.info(f"SHA!: {hashlib.sha1(f.read()).hexdigest()}")

    def load_model(self, model_file):
        file_path = Path("data_unversioned").joinpath(
            "models",
            self.args.tb_prefix,
            model_file + ".pt",
        )
        self.model.load_state_dict(torch.load(file_path))
        self.model.eval()

    def set_lr(self, i_epoch):
        """Set learning rate based on an exponential decay schedule
        between lr_initial and lr_final."""
        if i_epoch <= self.args.lr_decay_epochs:
            lr = self.args.lr_initial * (self.args.lr_final / self.args.lr_initial) ** (
                i_epoch / self.args.lr_decay_epochs
            )
            print(f"reducing lr: {lr}")
        else:
            lr = self.args.lr_final
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def train(self):
        pass


class SyntheticDemoTrainingApp(TrainingApp):
    """Top level application class for training AlphaTensor on synthetic data only."""

    def __init__(self):
        super().__init__()

    def init_dl(self):
        demos = SyntheticDemoDataset(
            self.args.dim_t,
            self.args.len_data,
            self.args.dim_t,
            self.args.dim_3d,
            self.args.device,
        )
        demos_train, demos_test = random_split(demos, [0.9, 0.1])
        dl_train = DataLoader(
            demos_train, batch_size=self.args.batch_size, shuffle=True
        )
        dl_val = DataLoader(demos_test, batch_size=self.args.batch_size, shuffle=True)
        return dl_train, dl_val

    def _take_action(self, state_batch, scalar_batch):
        """Input: current environment: state_batch and scalar_batch
        Output: new environment: state_batch and scalar_batch, best ranks, actions"""
        aa, pp, qq = self.model.fwd_infer(state_batch, scalar_batch, n_samples=1)
        uu, vv, ww = torch.split(aa.squeeze() - 2, self.args.dim_3d, dim=-1)
        action_tensor = uvw_to_tensor((uu, vv, ww))
        new_head = state_batch[:, 0] - action_tensor
        new_head = new_head.unsqueeze(dim=1)
        new_state_batch = torch.cat((new_head, state_batch), dim=1)
        new_state_batch = new_state_batch[:, :-1]
        grouped_samples = new_head.view(
            -1,
            self.args.n_samples,
            self.args.dim_3d,
            self.args.dim_3d,
            self.args.dim_3d,
        )
        rank_ubs = torch.sum(grouped_samples != 0, [-1, -2, -3], dtype=torch.int32)
        best_samples = torch.min(rank_ubs, -1)
        return new_state_batch, scalar_batch + 1, best_samples, (uu, vv, ww)

    def main(self):
        if self.args.model_file is not None:
            print(f"loading model {self.args.model_file}")
            self.load_model(self.args.model_file)
        dl_train, dl_val = self.init_dl()
        for i_epoch in range(self.args.n_epochs):
            self.set_lr(i_epoch)
            epoch_loss_pol = 0
            epoch_loss_val = 0
            # training epoch
            self.model.train()
            for state_batch, scalar_batch, action_batch, reward_batch in dl_train:
                print(
                    f"start train batch of size {state_batch.shape[0]}, train length {len(dl_train.dataset)}/{len(dl_train)}"
                )
                loss_pol, loss_val = self.model.fwd_train(
                    state_batch, scalar_batch, action_batch, reward_batch
                )
                epoch_loss_pol += loss_pol
                epoch_loss_val += loss_val
                loss_combined = (
                    self.args.weight_pol * loss_pol + self.args.weight_val * loss_val
                )
                self.optimizer.zero_grad()
                loss_combined.backward()
                self.optimizer.step()
            self.training_samples_count += len(dl_train.dataset)
            epoch_loss_pol /= len(dl_train.dataset)
            epoch_loss_val /= len(dl_train.dataset)
            self.log_metrics(i_epoch, "trn", epoch_loss_pol, epoch_loss_val)
            # train/validation loss printout
            if i_epoch % self.args.n_val == 0:
                print(
                    f"TRN epoch: {i_epoch} policy loss: {epoch_loss_pol} "
                    f"value loss {epoch_loss_val}"
                )
                self.model.eval()
                epoch_loss_pol = 0
                epoch_loss_val = 0
                for state_batch, scalar_batch, action_batch, reward_batch in dl_val:
                    loss_pol, loss_val = self.model.fwd_train(
                        state_batch, scalar_batch, action_batch, reward_batch
                    )
                    epoch_loss_pol += loss_pol
                    epoch_loss_val += loss_val
                epoch_loss_pol /= len(dl_val.dataset)
                epoch_loss_val /= len(dl_val.dataset)
                self.log_metrics(i_epoch, "val", epoch_loss_pol, epoch_loss_val)
                print(
                    f"VAL epoch: {i_epoch} policy loss: {epoch_loss_pol} "
                    f"value loss {epoch_loss_val}"
                )
            if i_epoch % self.args.n_save == 0:
                self.save_model("synth", i_epoch)
            # Solution search printout
            if i_epoch % self.args.n_act == 0:
                for dl, val in [(dl_train, "train"), (dl_val, "val")]:
                    print(val)
                    self.model.eval()
                    lowest_rank = torch.tensor(self.model.dim_3d**3)
                    num_solutions_found = 0
                    for state_batch, scalar_batch, _, _ in dl:
                        state_batch = state_batch.repeat(
                            self.args.n_samples, 1, 1, 1, 1
                        )
                        scalar_batch = scalar_batch.repeat(self.args.n_samples, 1)
                        for i in range(self.args.max_actions):
                            (
                                state_batch,
                                scalar_batch,
                                best_samples,
                                _,
                            ) = self._take_action(state_batch, scalar_batch)
                            lowest_rank = torch.min(
                                lowest_rank, torch.min(best_samples.values)
                            )
                            num_solutions_found += torch.sum(best_samples.values == 0)
                    if num_solutions_found > 0:
                        print(
                            f"E{i_epoch}: Found {num_solutions_found} solutions out of {len(dl.dataset)}"
                        )
                    else:
                        print(f"E{i_epoch} : lowest rank found = {lowest_rank}")


class TensorGameTrainingApp(TrainingApp):
    """Top level application class for training AlphaTensor on a tensor game dataset"""

    def __init__(self):
        super().__init__()
        self.dataset, self.dataset_val = self.init_ds()
        self.dl = self.init_dl()

    def init_ds(self):
        if self.args.start_rank:
            # Create a start_tensor from a synthetic demo
            values = torch.tensor((-1, 0, 1))
            probs = torch.tensor((0.1, 0.8, 0.1))
            shift = 1
            action_seq, start_tensor = create_synthetic_demo(
                values,
                probs,
                self.args.start_rank,
                self.args.dim_3d,
                shift,
            )
            # test case
            # action_seq = [torch.tensor([2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1])]
            # start_tensor = action_to_tensor(action_seq[0])
            start_tensor = start_tensor.unsqueeze(0)
            start_tensor = torch.cat(
                (
                    start_tensor,
                    torch.zeros(
                        (
                            self.args.dim_t - 1,
                            self.args.dim_3d,
                            self.args.dim_3d,
                            self.args.dim_3d,
                        )
                    ),
                )
            )
        else:
            start_tensor = None
        dataset = TensorGameDataset(
            self.args.len_data,
            self.args.fract_synth,
            self.args.max_actions,
            self.args.dim_t,
            self.args.dim_3d,
            self.args.device,
            start_tensor=start_tensor,
            action_seq=action_seq,
        )
        dataset_val = SyntheticDemoDataset(
            self.args.max_actions,
            2000,
            self.args.dim_t,
            self.args.dim_3d,
            self.args.device,
            save_dir=SAVE_DIR_VAL,
        )
        return dataset, dataset_val

    def init_dl(self):
        dl = DataLoader(self.dataset, batch_size=self.args.batch_size, shuffle=True)
        return dl

    def train_step(self, i_epoch):
        """Train model on up to 5 batches"""
        self.set_lr(i_epoch)
        self.dataset.resample_buffer_indexes()
        self.model.train()
        dl = DataLoader(self.dataset, batch_size=self.args.batch_size, shuffle=True)
        epoch_loss_pol = 0
        epoch_loss_val = 0
        for state_batch, scalar_batch, action_batch, reward_batch in dl:
            # state_batch ~ (batch_size, dim_t, dim_3d, dim_3d, dim_3d)
            # scalar_batch ~ (batch_size, 1)
            # action_batch ~ (batch_size, n_steps)
            # reward_batch ~ (batch_size, 1)
            loss_pol, loss_val = self.model.fwd_train(
                state_batch, scalar_batch, action_batch, reward_batch
            )
            loss = self.args.weight_pol * loss_pol + self.args.weight_val * loss_val
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss_pol += loss_pol
            epoch_loss_val += loss_val
        epoch_loss_pol /= len(dl.dataset)
        epoch_loss_val /= len(dl.dataset)
        self.training_samples_count += len(dl.dataset)
        self.log_metrics(i_epoch, "trn", epoch_loss_pol, epoch_loss_val)

    @torch.no_grad()
    def val_step(self, i_epoch):
        self.model.eval()
        epoch_loss_pol = 0
        epoch_loss_val = 0
        dl = DataLoader(self.dataset_val, batch_size=self.args.batch_size)
        for state_batch, scalar_batch, action_batch, reward_batch in dl:
            loss_pol, loss_val = self.model.fwd_train(
                state_batch, scalar_batch, action_batch, reward_batch
            )
            epoch_loss_pol += loss_pol
            epoch_loss_val += loss_val
        epoch_loss_pol /= len(dl.dataset)
        epoch_loss_val /= len(dl.dataset)
        self.log_metrics(i_epoch, "val", epoch_loss_pol, epoch_loss_val)

    @torch.no_grad()
    def act_step(self):
        """Plays a set of games and adds trajectories to played games buffer.
        Best game is added to best games buffer.
        """
        self.model.eval()
        best_reward = -1e6
        best_game = None
        for _ in range(self.args.n_games):
            state_seq, action_seq, reward_seq = actor_prediction(
                self.model,
                self.dataset.start_tensor,
                self.args.max_actions,
                self.args.n_sim,
                self.args.n_bar,
            )
            if reward_seq[-1] > best_reward:
                best_reward = reward_seq[-1]
                best_game = (state_seq, action_seq, reward_seq)
            self.dataset.add_played_game(state_seq, action_seq, reward_seq)
        if best_game is not None:
            self.dataset.add_best_game(*best_game)
            self.val_writer.add_scalar(
                "best reward", best_reward, self.training_samples_count
            )
            log.info(f"best_reward  {best_reward}")

    def main(self):
        self.model = self.model.to(self.args.device)
        print_params(self.model)
        self.dataset.set_fractions(self.args.fract_synth, self.args.fract_best)
        for i_epoch in range(self.args.n_epochs):
            if i_epoch + 1 == self.args.n_epochs // 50:
                self.dataset.set_fractions(0.25, 0.05)
            t0 = time.time()
            self.train_step(i_epoch)
            t1 = time.time()
            print(f"train time {t1 - t0}")

            # Validate
            if i_epoch % self.args.n_val == 0:
                t0 = time.time()
                self.val_step(i_epoch)
                t1 = time.time()
                print(f"val time {t1 - t0}")

            # Search for solution
            if i_epoch % self.args.n_act == 0:
                t0 = time.time()
                self.act_step()
                t1 = time.time()
                print(f"act time {t1 - t0}")

            # Save model
            if i_epoch % self.args.n_save == 0:
                self.save_model(self.args.tb_prefix, i_epoch)


if __name__ == "__main__":
    TensorGameTrainingApp().main()
