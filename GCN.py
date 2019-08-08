#!/usr/bin/env python
# coding: utf-8

# ## Dataset Class

# In[4]:


import os
import os.path as osp

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric.utils import from_networkx

class MaxIndDataset(InMemoryDataset):
    def __init__(self, root):
        self.label_frame = pd.read_csv(osp.join(root, "label.csv"))
        self.weight_frame = pd.read_csv(osp.join(root, "weight.csv"))
        self.root_dir = root
        self.num_graphs = len(self.label_frame)
        super(MaxIndDataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])
   
    @property   
    def processed_file_names(self):
        return ['data.pt']
    
    def _download(self):
        pass
    
    def process(self):
        data_list = []
        for i in range(self.num_graphs):
            graph_name = os.path.join(self.root_dir, self.label_frame.iloc[i, 0])
            graph = nx.read_adjlist(graph_name, nodetype=int)
   
            weight = torch.tensor(self.weight_frame.iloc[i, 1:], dtype=torch.float)
            label = torch.tensor(self.label_frame.iloc[i, 1:], dtype=torch.long)
        
            data = from_networkx(graph)
            data.x = weight
            data.y = label
            
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


# ## Drawing Utility
# 
# Takes in a data entry:
# * "edge_list" (list of edges)
# * "x" labels (weights)
# * "y" lables (output labels)
# 
# and draws the graph out in a circular format. 

# In[55]:


from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import torch
def draw_entry(entry, color_map = None, title = None, node_dict = None):
    g = to_networkx(entry)
    
    if color_map:
        label = color_map
    else:
        label = entry["y"]
    
    # Create color map from selected nodes, green for selected, grey for unselected.
    color_map = ["grey"] * len(g.nodes)

    for i in np.flatnonzero(label):
        color_map[i] = "green"

    node_labels = entry["x"]

    if node_dict is None:
        node_dict = {v:v for v in range(node_labels)}

    if not torch.equal(node_labels, torch.ones(len(g.nodes))):
        node_labels = {k: "{0}:\n{1:.3f}".format(node_dict[k], v) for (k, v) in enumerate(node_labels)}
    else:
        node_labels = {k: node_dict[k] for k in g.nodes}
    
    plt.figure()
    plt.title(title)
    pos = nx.circular_layout(g)
    nx.draw(g, pos, node_size=2000, width = 1, node_color = color_map)
    nx.draw_networkx_labels(g, pos, node_labels)
    plt.show()  
    





def split_loader(dataset, train_size, test_size, batch_size):
    dataset.shuffle()
    size = len(dataset)
    
    tr_i = int(size * train_size)
    val_i = tr_i + int(size * test_size)
    train_loader = DataLoader(dataset[:tr_i], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset[tr_i: val_i])
    test_loader = DataLoader(dataset[val_i:])
    return train_loader, val_loader, test_loader

# In[6]:
if __name__ == '__main__':

    visualize = MaxIndDataset('data/mini')


    # In[11]:


    draw_entry(visualize[2])


    # In[12]:


    bin_80 = MaxIndDataset('data/binomial_80')


    # In[13]:


    train_loader, val_loader, test_loader = split_loader(bin_80, .7, .2, 20)


    # In[14]:



    for b in train_loader:
        print(b)
        print(b.x.size())
        print(b.edge_index.size())
        break


    # In[15]:


    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv

    class BasicNet(torch.nn.Module):
        def __init__(self):
            super(BasicNet, self).__init__()
            self.conv1 = GCNConv(1, 16)
            self.conv2 = GCNConv(16, 1)

        def forward(self, data):
            weight = data["x"].view(-1, 1)
            edge_index = data["edge_index"]
            x = self.conv1(weight, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            x = torch.sigmoid(x)
            return x


    # In[16]:


    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv

    class MultiNet(torch.nn.Module):
        def __init__(self):
            super(MultiNet, self).__init__()
            self.conv_in = GCNConv(1, 32)
            self.conv_mid = GCNConv(32, 32)
            self.conv_out = GCNConv(32, 1)

        def forward(self, data):
            weight = data["x"].view(-1, 1)
            edge_index = data["edge_index"]
            x = self.conv_in(weight, edge_index)
            x = F.relu(x)

            x = self.conv_mid(x, edge_index)
            x = F.relu(x)

            x = self.conv_mid(x, edge_index)
            x = F.relu(x)

            x = self.conv_mid(x, edge_index)
            x = F.relu(x)

            x = self.conv_mid(x, edge_index)
            x = F.relu(x)

            x = self.conv_out(x, edge_index)
            x = torch.sigmoid(x)
            return x


    # In[42]:


    from tqdm import tqdm#, tqdm_notebook

    ### Runs but losses dont go down, seems to converge to around 0.48

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = MultiNet().to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    def train(epoch):
        model.train()
        running_loss = 0.0
        for i, item in enumerate(train_loader):
            optimizer.zero_grad()
            item = item.to(device)
            outputs = model(item)

            loss = criterion(outputs, item["y"].float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        return running_loss / len(train_loader)

    def evaluate(loader):
        model.eval()
        running_acc = 0.0 # Implement later
        running_loss = 0.0
        for i, item in enumerate(loader):
            item = item.to(device)
            outputs = model(item)
            loss = criterion(outputs.view(-1, 1), item["y"].float().view(-1, 1))
            running_loss += loss.item()
        return running_loss / len(loader)

    for epoch in tqdm(range(15)):
        train_loss = train(epoch)
        val_loss = evaluate(val_loader)
        print(('Epoch: {:03d}, Train Loss: {:.4f}, Val Loss: {:.3f}').format(epoch, train_loss, val_loss))

    print('Finished Training')


    # In[44]:


    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())


    # In[45]:


    torch.save(model.state_dict(), "model/unweighted_multinet.pt")


    # In[46]:


    model.eval()


    # In[47]:


    test_item = visualize[2]
    draw_entry(test_item)


    # In[48]:


    test_item.to(device)
    out = model(test_item)
    out


    # In[49]:


    # Sorts the indices from greatest value to the least
    ans = np.flip(out.detach().cpu().numpy().T.argsort())
    ans


    # In[57]:


    edges = test_item["edge_index"].cpu().numpy().T
    edges


    # In[90]:


    for e in edges:
        print(e)


    # In[78]:


    edges[:, 0] != 3


    # In[79]:


    edges[:, 1] != 3


    # In[81]:


    np.logical_and(edges[:, 0] != 3, edges[:, 1] != 3)


    # In[ ]:




