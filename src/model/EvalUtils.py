import torch

from env.Env import MaxSetState
from data.dataset import MaxIndDataset
from model.GCN import ConvNet3
import os.path as osp

def max_eval(tensor_item, model):
    state = MaxSetState.loadFromTensor(tensor_item)
    tot_reward = 0
    while len(state.actions()) > 0:
        item = state.getTensorItem()
        probs = model(item)
        action = state.max_action(probs.cpu().detach().numpy())
        state, reward = state.step(action)
        tot_reward += reward
    return tot_reward


def prob_eval(tensor_item, model):
    state = MaxSetState.loadFromTensor(tensor_item)
    tot_reward = 0
    while len(state.actions()) > 0:
        item = state.getTensorItem()
        probs = model(item)
        action = state.prob_action(probs.cpu().detach().numpy())
        state, reward = state.step(action)
        tot_reward += reward
    return tot_reward


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conv_net = ConvNet3().to(device)
    name = "weighted_convnet_epoch_19.pt"
    folder = "wandb/run-20190821_131619-t00lpy2e/"
    params = torch.load(osp.join(folder, name))
    conv_net.load_state_dict(params)

    visualize = MaxIndDataset("../../data/weighted_mix_2")

    for i in range(len(visualize)):
        tot_reward = max_eval(visualize[i], conv_net)