import torch
import os.path as osp
# import GCL.losses as L

from torch import nn
from tqdm import tqdm
from torch.optim import Adam, SGD
# from GCL.eval import get_split, SVMEvaluator
# from GCL.models import SingleBranchContrast
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
import pickle
import gc

device = torch.device("cuda:4")
def make_gin_conv(input_dim, out_dim):
    return GINConv(nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))


class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers):
        super(GConv, self).__init__()
        self.activation = activation()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(make_gin_conv(input_dim, hidden_dim))
            else:
                self.layers.append(make_gin_conv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, batch):
        z = x
        zs = []
        for conv, bn in zip(self.layers, self.batch_norms):
            z = conv(z, edge_index)
            z = self.activation(z)
            z = bn(z)
            zs.append(z)
        gs = [global_add_pool(z, batch) for z in zs]
        z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
        return z, g


class FC(nn.Module):
    def __init__(self, hidden_dim):
        super(FC, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return self.fc(x) + self.linear(x)


class Encoder(torch.nn.Module):
    def __init__(self, encoder, local_fc, global_fc):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.local_fc = local_fc
        self.global_fc = global_fc

    def forward(self, x, edge_index, batch):
        z, g = self.encoder(x, edge_index, batch)
        return z, g

    def project(self, z, g):
        return self.local_fc(z), self.global_fc(g)


def train(encoder_model, contrast_model, dataloader, optimizer):
    encoder_model.train()
    epoch_loss = 0
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()

        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

        z, g = encoder_model(data.x, data.edge_index, data.batch)
        z, g = encoder_model.project(z, g)
        loss = contrast_model(h=z, g=g, batch=data.batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss

def main():
    device = torch.device(torch.device("cuda:4"))
    dataset = pickle.load(open('graph_data_chembl.pkl', 'rb'))
    dataloader = DataLoader(dataset, batch_size=256)
    gconv = GConv(input_dim=9, hidden_dim=64, activation=torch.nn.ReLU, num_layers=3).to(device)
    fc1 = FC(hidden_dim=64 * 3)
    fc2 = FC(hidden_dim=64 * 3)
    encoder_model = Encoder(encoder=gconv, local_fc=fc1, global_fc=fc2).to(device)
    # contrast_model = SingleBranchContrast(loss=L.JSD(), mode='G2L').to(device)

    optimizer = Adam(encoder_model.parameters(), lr=0.01)
    min_loss = 10e5
    loss_hst = []
    with tqdm(total=100, desc='(T)') as pbar:
        for epoch in range(1, 200):
            loss = train(encoder_model, contrast_model, dataloader, optimizer)
            pbar.set_postfix({'loss': loss})
            pbar.update()
            loss_hst.append(loss)
            filehandler = open(f"lost_hst.pkl","wb") 
            pickle.dump(loss_hst,filehandler)
            if loss < min_loss:
                min_loss = loss
                torch.save(encoder_model.state_dict(), 'encoder_best.pt')
                torch.save(contrast_model.state_dict(), 'contrast_best.pt')
            gc.collect()

