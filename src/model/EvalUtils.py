import torch

from env.Env import MaxSetState
from data.dataset import KidneyDataset
from model.GCN import ConvNet3
import os.path as osp

import math
import time
from tqdm import tqdm
def max_eval(tensor_item, model):
    state = MaxSetState.loadFromTensor(tensor_item)
    tot_reward = 0
    actions = []

    tot_time = 0
    iters = 0
    while len(state.actions()) > 0:
        item = state.getTensorItem()

        probs = model(item)
        action = state.max_action(probs.cpu().detach().numpy())

        actions.append(action)
        init_t = time.time()
        state, reward = state.step(action)
        end_t = time.time()
        tot_reward += reward
        iters += 1
        tot_time += end_t - init_t

    #print("Time: {0} / {1}".format(tot_time, iters))
    checked_reward = sum(tensor_item["x"][actions]).item()
    if not math.isclose(checked_reward, tot_reward):
        raise Exception("Reward is not matching! {0} != {1}".format(tot_reward, checked_reward))
    return tot_reward, actions


def prob_eval(tensor_item, model):
    state = MaxSetState.loadFromTensor(tensor_item)
    tot_reward = 0
    actions = []
    while len(state.actions()) > 0:
        item = state.getTensorItem()
        probs = model(item)
        action = state.prob_action(probs.cpu().detach().numpy())
        state, reward = state.step(action)
        tot_reward += reward
        actions.append(action)
    return tot_reward, actions


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conv_net = ConvNet3().to(device)
    name = "weighted_convnet_epoch_19.pt"
    folder = "wandb/run-20190821_131619-t00lpy2e/"
    params = torch.load(osp.join(folder, name))
    conv_net.load_state_dict(params)

    visualize = KidneyDataset("../../data/weighted_mix_2")

    for i in range(len(visualize)):
        tot_reward = max_eval(visualize[i], conv_net)