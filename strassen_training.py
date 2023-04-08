from data_generation import *
from model import *
from training import TrainingApp

from torch.utils.data import DataLoader


class StrassenTrainingApp(TrainingApp):
    def __init__(self):
        super().__init__()
        self.dl = self.init_dl()

    def init_dl(self):
        strassen = StrassenDemoDataset(max_len=self.args.max_len)
        dl = DataLoader(strassen, batch_size=strassen.n_demos, shuffle=True)
        return dl

    def _play_strassen(self, dl, max_actions=7):
        self.model.eval()
        strassen_tensor, action_list = get_strassen_tensor("cpu")
        strassen_input = strassen_tensor.unsqueeze(0).unsqueeze(0)
        current_state = strassen_input.clone()
        scalar_fixed = torch.tensor([[0.0]])
        print(
            f"start play : {torch.sum(strassen_input != 0)} nonzero elements remaining"
        )
        for ii in range(max_actions):
            for st in dl.dataset.state_tensor:
                if torch.all(torch.eq(st, current_state.squeeze(0))):
                    print("Current state found in dataset")
                    break
            aa, pp, qq = self.model.fwd_infer(current_state, scalar_fixed, n_samples=1)
            valid = False
            for j in range(action_list.shape[0]):
                if torch.all(torch.eq(aa.squeeze(), action_list[j])):
                    valid = j
                    break
            print(f"action: {aa.squeeze()} ; valid: {valid}")
            uu, vv, ww = torch.split(aa.squeeze() - 2, 4, dim=-1)
            action_tensor = factors_to_tensor((uu, vv, ww))
            current_state = current_state - action_tensor
            print(
                f"step {ii} : {torch.sum(current_state != 0)} nonzero elements remaining"
            )
            if torch.sum(current_state != 0) == 0:
                print(f"************* Produced Strassen factorization *************")
                break

    def main(self):
        dl = self.init_dl()
        for i_epoch in range(self.args.n_epochs):
            epoch_loss_pol = 0
            epoch_loss_val = 0
            for states, scalars, target_actions, rewards in dl:
                self.model.train()
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
                if i_epoch % self.args.n_print == 0:
                    print(
                        f"epoch: {i_epoch} policy loss: {epoch_loss_pol} value loss {epoch_loss_val}"
                    )
                    self.model.eval()
                    aa, pp, qq = self.model.fwd_infer(
                        states, scalars, n_samples=self.args.n_samples
                    )
                    correct = torch.eq(aa.transpose(0, 1), target_actions)
                    print(f"Percent correct= {100*torch.mean(correct.float())}")
                    print(f"Baseline= {100*torch.mean((target_actions == 2).float())}")
                    pass
                if i_epoch % self.args.n_act == 0:
                    self._play_strassen(dl)


def confirm_strassen(strassen):
    strassen_tensor, action_list = get_strassen_tensor("cpu")
    strassen_input = strassen_tensor.unsqueeze(0).unsqueeze(0)
    current_state = strassen_input.clone()
    print(f"start : {torch.sum(current_state != 0)} nonzero elements remaining")
    for ii in range(7):
        for st in strassen.state_tensor:
            if torch.all(torch.eq(st, current_state.squeeze(0))):
                print("Current state found in dataset")
                break
        ff = action_list[ii]
        uu, vv, ww = torch.split(ff - 2, 4, dim=-1)
        action_tensor = factors_to_tensor((uu, vv, ww))
        current_state = current_state - action_tensor
        print(f"step {ii} : {torch.sum(current_state != 0)} nonzero elements remaining")


if __name__ == "__main__":
    StrassenTrainingApp().main()
