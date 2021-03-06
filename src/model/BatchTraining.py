from collections import OrderedDict
from multiprocessing import Process

from tqdm import tqdm#, tqdm_notebook
from model.GCN import *
from data.dataset import *
from model import EvalUtils
import os
import os.path as osp
from model.EvalUtils import *

import wandb
import torch.multiprocessing as mp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.BCELoss()
def train(model, optimizer, loader):
    model.train()
    running_loss = 0.0
    for batch_idx, item in (enumerate(loader)):
        optimizer.zero_grad()
        item = item.to(device)
        outputs = model(item)
        loss = criterion(outputs, item["y"].float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def ce_eval(model, loader):
    model.eval()
    running_acc = 0.0 # Implement later
    running_loss = 0.0
    for i, item in enumerate(loader):
        item = item.to(device)
        outputs = model(item)
        loss = criterion(outputs.view(-1, 1), item["y"].float().view(-1, 1))
        running_loss += loss.item()
    return running_loss / len(loader)

def score_eval(model, loader):
    model.eval()
    running_score = 0
    for i, item in enumerate(loader):
        node_weights = item["x"].detach().cpu().numpy().flatten()
        y = item["y"].detach().cpu().numpy().flatten().astype(bool)
        opt_score = sum(node_weights[y])
        ratio = EvalUtils.prob_eval(item, model) / opt_score

        if ratio > 1.001:
            raise Exception("Ratio is {0} > 1".format(ratio))
        running_score += ratio
    return running_score / len(loader)

import random
import string

VOWELS = "aeiou"
CONSONANTS = "".join(set(string.ascii_lowercase) - set(VOWELS))

def generate_word(length):
    word = ""
    for i in range(length):
        if i % 2 == 0:
            word += random.choice(CONSONANTS)
        else:
            word += random.choice(VOWELS)
    return word

if __name__ == '__main__':

    SAVE_FREQ = 5

    model_name = "deepconvnet_2"
    project_name = "kidney"
    run_folder = "../../model/{0}/".format(project_name)

    if not osp.exists(run_folder):
        print("Specified checkpoint directory does not exist! Making directory [{}].".format(run_folder))
        os.mkdir(run_folder)
    wandb.init(name="{0}_{1}".format(model_name, generate_word(5)), project=project_name, entity="deepkidney", reinit=True)
    model = MultiNet().to(device)
    wandb.watch(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0008, weight_decay=5e-6)
    training_set = KidneyDataset('../../data/kidney_train/weighted_n500_p15')
    train_loader, val_loader, test_loader = split_loader(training_set, .8, .15, 25)
    for epoch in tqdm(range(25)):
        train_loss = train(model, optimizer, train_loader)
        val_loss = ce_eval(model, val_loader)
        #score = score_eval(model, test_loader)

        # tqdm.write(
        #     ('Epoch: {0:03d}, Train Loss: {1:.4f}, Val Loss: {2:.3f}, Score: {3:.3f}').format(epoch, train_loss,
        #                                                                                       val_loss, score))
        wandb.log({"Train Loss": train_loss, "Val Loss": val_loss})

        if epoch % SAVE_FREQ == SAVE_FREQ - 1:
            folder = osp.join(wandb.run.dir, "model/")
            name = "epoch_{0}.pt".format(epoch)
            wandb_fp = osp.join(folder, name)
            if not osp.exists(folder):
                #print("Specified checkpoint directory does not exist! Making directory [{}].".format(folder))
                os.mkdir(folder)
            wandb.save(wandb_fp)

            folder = osp.join(run_folder, "size_{0}".format(model_name))
            if not osp.exists(folder):
                #print("Specified checkpoint directory does not exist! Making directory [{}].".format(folder))
                os.mkdir(folder)
            torch_fp = osp.join(folder, name)
            torch.save(model.state_dict(), torch_fp)