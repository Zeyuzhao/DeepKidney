from collections import OrderedDict
from multiprocessing import Process

from tqdm import tqdm#, tqdm_notebook
from model.GCN import *
from data.dataset import *
from model import EvalUtils
import os
import os.path as osp


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

        if ratio > 1:
            raise Exception("Ratio is {0} > 1".format(ratio))
        running_score += ratio
    return running_score / len(loader)


SAVE_FREQ = 5

if __name__ == '__main__':
    for i, num_nodes in enumerate(range(10, 101, 10)):
        wandb.init(name="size_{0}".format(num_nodes), project="gcn", entity="deepkidney", reinit=True)
        model = ConvNet3().to(device)
        wandb.watch(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0008, weight_decay=5e-6)

        training_set = MaxIndDataset('../../data/mixed_42/weighted_{0}'.format(num_nodes))
        train_loader, val_loader, test_loader = split_loader(training_set, .8, .15, 25)
        for epoch in tqdm(range(10)):
            train_loss = train(model, optimizer, train_loader)
            val_loss = ce_eval(model, val_loader)
            score = score_eval(model, test_loader)

            # tqdm.write(
            #     ('Epoch: {0:03d}, Train Loss: {1:.4f}, Val Loss: {2:.3f}, Score: {3:.3f}').format(epoch, train_loss,
            #                                                                                       val_loss, score))
            wandb.log({"Train Loss": train_loss, "Val Loss": val_loss, "Score:": score})

            if epoch % SAVE_FREQ == SAVE_FREQ - 1:
                folder = osp.join(wandb.run.dir, "model/")
                name = "weighted_convnet_epoch_{0}.pt".format(epoch)
                fp = osp.join(folder, name)
                if not osp.exists(folder):
                    print("Specified checkpoint directory does not exist! Making directory [{}].".format(folder))
                    os.mkdir(folder)
                torch.save(model.state_dict(), fp)
                wandb.save(fp)