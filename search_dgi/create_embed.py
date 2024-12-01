import torch
import os.path as osp
# import GCL.losses as L

from torch import nn
from tqdm import tqdm
from torch.optim import Adam
# from GCL.eval import get_split, SVMEvaluator
# from GCL.models import SingleBranchContrast
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.loader import DataLoader
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
        print(z.shape)
        print(g.shape)
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
    loss_hst   = []
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        z, g = encoder_model(data.x, data.edge_index, data.batch)
        z, g = encoder_model.project(z, g)
        loss = contrast_model(h=z, g=g, batch=data.batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        loss_hst.append(loss.item())
    return epoch_loss, loss_hst

def inference(encoder_model, dataloader):
    encoder_model.eval()
    # index_ligands = 0
    embed_lst = []
    for data in dataloader:
        data = data.to('torch.device("cuda:6")')
        z, g = encoder_model(data.x, data.edge_index, data.batch)
        z, g = encoder_model.project(z, g)
        for i in torch.unique(data.batch):
            # if index_ligands <= 1175392:
            #     index_ligands += 1
            #     continue
            # filehandler = open(f"./potential_embed/{index_ligands}.pkl","wb") 
            # pickle.dump(torch.sum(torch.cat((z[data.batch == i], g[i].repeat(torch.sum(data.batch == i), 1)), dim = 1), dim = 0),filehandler)
            # index_ligands += 1
            # embed_lst.append(torch.sum(torch.cat((z[data.batch == i], g[i].repeat(torch.sum(data.batch == i), 1)), dim = 1), dim = 0))
            embed_lst.append(g[i])
        filehandler = open(f"./graph_data_5dl2_potential_ligands_embed.pkl","wb") 
        pickle.dump(embed_lst,filehandler)
        filehandler.close()
def main():
    
    device = torch.device(device)
    dataset = pickle.load(open('./graph_data_5dl2_potential_ligands.pkl', 'rb'))
    dataloader = DataLoader(dataset, batch_size=512, shuffle = False)
    gconv = GConv(input_dim=9, hidden_dim=64, activation=torch.nn.ReLU, num_layers=3).to(device)
    fc1 = FC(hidden_dim=64 * 3)
    fc2 = FC(hidden_dim=64 * 3)
    encoder_model = Encoder(encoder=gconv, local_fc=fc1, global_fc=fc2).to(device)
    encoder_model.load_state_dict(torch.load("encoder_best.pt"))
    encoder_model.eval()
    inference(encoder_model,dataloader)


if __name__ == '__main__':
    main()