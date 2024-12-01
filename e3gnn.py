import e3gnn_utils as eg
import torch
import sys
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import set_seed
import subprocess
import shutil
import warnings
import pdbreader
from torch_geometric.nn import GATConv
import torch_geometric
import torch
from torch_geometric.nn import GATConv
from sklearn.metrics.pairwise import cosine_similarity
import math
from glob import glob
import os
import pandas as pd
import sys
from utils import *
import torch
from egnn_pytorch import EGNN_Network
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

device_num = 6
net = EGNN_Network(
    num_tokens = 100,
    dim = 64,
    depth = 3,
    only_sparse_neighbors = True
).cuda(device_num)

class E3GNNDataloader(Dataset):
    def __init__(self, atoms_path = 'drugbank_atoms.pt', bonds_path = 'drugbank_bonds.pt', coords_path = 'drugbank_coords.pt', mask_path='drugbank_masks.pt'):
        self.drugbank_atoms  = torch.load(atoms_path)
        self.drugbank_bonds  = torch.load(bonds_path)
        self.drugbank_coords = torch.load(coords_path)
        self.drugbank_masks  = torch.load(mask_path)

    def __len__(self):
        return len(self.drugbank_atoms)

    def __getitem__(self, idx):
        return self.drugbank_atoms[idx].cuda(device_num), self.drugbank_bonds[idx].cuda(device_num), self.drugbank_coords[idx].cuda(device_num), self.drugbank_masks[idx].cuda(device_num)


dataloader = E3GNNDataloader()
train_dataloader = DataLoader(dataloader, batch_size=256, shuffle=False)
i = 0
data_lst = []
for sample in train_dataloader:
    feats_out, coords_out = net(sample[0].int(), sample[2].float(), mask = sample[3].bool(), adj_mat = sample[1].bool())
    data_lst.append(torch.mean(feats_out, dim = 2).detach().cpu())

data_embed = torch.concatenate(data_lst)
torch.save(data_embed, 'drugbank_egnn.pt')
paths = glob('drug_disease/*')
for path in paths:
    protein_name = os.path.split(path)[1]
    protein_path = f'drug_disease/{protein_name}/'
    equipath = os.path.join(protein_path, 'e3gnn')
    if not os.path.exists(equipath):
        continue
    generate_atoms  = f'drug_disease/{protein_name}/e3gnn/e3gnn_generate_atoms.pt'
    generate_bonds  = f'drug_disease/{protein_name}/e3gnn/e3gnn_generate_bonds.pt'
    generate_coords = f'drug_disease/{protein_name}/e3gnn/e3gnn_generate_coords.pt'
    generate_masks  = f'drug_disease/{protein_name}/e3gnn/e3gnn_generate_masks.pt'

    dataloader = E3GNNDataloader(generate_atoms,generate_bonds,generate_coords,generate_masks)
    train_dataloader = DataLoader(dataloader,batch_size=256, shuffle=False)
    i = 0
    generate_lst = []
    for sample in train_dataloader:
        feats_out, coords_out = net(sample[0].int(), sample[2].float(), mask = sample[3].bool(), adj_mat = sample[1].bool())
        generate_lst.append(torch.mean(feats_out, dim = 2).detach().cpu())

    generate_embed = torch.concatenate(generate_lst)

    egnn_path = os.path.join(protein_path, 'egnn')
    if not os.path.exists(egnn_path):
        os.makedirs(egnn_path)
    torch.save(generate_embed, os.path.join(egnn_path, 'e3gnn_embed.pt'))
    print(os.path.join(egnn_path, 'e3gnn_embed.pt'))


drugbank_equiform = torch.load('drugbank_egnn.pt')
drugbank_df = pd.read_csv('datasets/drugbank.csv')
paths = glob('drug_disease/*')
for path in paths:
    protein_name = os.path.split(path)[1]
    protein_path = f'drug_disease/{protein_name}/'
    egnnpath = os.path.join(protein_path, 'egnn')
    generative_path = os.path.join(egnnpath, 'e3gnn_embed.pt')
    if not os.path.exists(generative_path):
        continue
    generative_embed = torch.load(generative_path)
    cosine_similarity_matrix = cosine_similarity(drugbank_equiform, generative_embed)
    masks_valid =  torch.load(os.path.join(equipath, 'masks_valid.pt'))
    wrong_index = np.where(np.array(masks_valid) == 0)
    cosine_similarity_matrix[:,wrong_index] = 0
    argmax_ind = np.argmax(cosine_similarity_matrix, axis = 1)
    generation_docking_path = os.path.join(protein_path,'generation_docking/generation_docking.csv')
    generation_df = pd.read_csv(generation_docking_path)

    xs = []
    ys = []
    zs = []
    equi_scores = []
    bas         = []
    for drug_id in range(len(drugbank_df)):
        xs.append(generation_df.iloc[argmax_ind[drug_id]]['x'])
        ys.append(generation_df.iloc[argmax_ind[drug_id]]['y'])
        zs.append(generation_df.iloc[argmax_ind[drug_id]]['z'])
        bas.append(generation_df.iloc[argmax_ind[drug_id]]['ba'])
        equi_scores.append(cosine_similarity_matrix[drug_id, argmax_ind[drug_id]])
    drugbank_df['x'] = xs
    drugbank_df['y'] = ys
    drugbank_df['z'] = zs
    drugbank_df['ba'] = bas
    drugbank_df['sim_score'] = equi_scores
    bas = np.array(bas)
    equi_scores = np.array(equi_scores)
    E_equi_scores_squared = math.sqrt(np.mean(equi_scores**2))
    E_bas_squared = math.sqrt(np.mean(bas**2))
    final_score = ((2/3) * equi_scores/E_equi_scores_squared) - ((1/3) * bas/(E_bas_squared))
    drugbank_df['final_score'] = final_score
    sorted_df_multi = drugbank_df.sort_values(by=['final_score'], ascending=[False])
    sorted_df_multi['index'] = np.arange(len(sorted_df_multi))
    sorted_df_multi.to_csv(egnnpath + '/egnnscore.csv')

paths = glob('drug_disease/*')
for path in paths:
    protein_name = os.path.split(path)[1]
    protein_path = f'drug_disease/{protein_name}/'
    egnnpath = os.path.join(protein_path, 'egnn')
    csv_path = egnnpath + '/egnnscore.csv'
    if os.path.exists(csv_path):
        shutil.copy(csv_path, f'egnn/{protein_name}.csv')

