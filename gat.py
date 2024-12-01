import os
import sys
from utils import *
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
warnings.filterwarnings("ignore")

class GATWithEdgeAttr(torch.nn.Module):
    def __init__(self, in_channels, out_channels, edge_attr_dim):
        super(GATWithEdgeAttr, self).__init__()
        # Define two GATConv layers with edge attributes
        self.gat1 = GATConv(in_channels, 8, heads=8, edge_dim=edge_attr_dim)
        self.gat2 = GATConv(8 * 8, out_channels, heads=1, edge_dim=edge_attr_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        # Apply the first GATConv layer
        x = self.gat1(x, edge_index, edge_attr)
        x = torch.nn.functional.elu(x)
        # Apply the second GATConv layer
        x = self.gat2(x, edge_index, edge_attr)
        # Perform global pooling to combine node embeddings into a graph-level embedding
        x = torch_geometric.nn.global_mean_pool(x, batch)
        return x


dataset_path            =       'datasets/drugbank.csv'
paths = glob('drug_disease/*')
for path in paths:
    protein_name = os.path.split(path)[1]
    path_generate_ligand    =       f'drug_disease/{protein_name}/generation_docking/generation_docking.csv'
    if not os.path.exists(path_generate_ligand):
        continue
    dataset_df  = pd.read_csv(dataset_path)
    generate_df = pd.read_csv(path_generate_ligand)

    dataset_smiles  = []
    generate_smiles = []
    for i in range(len(dataset_df["smiles"])):
        dataset_smiles.append(get_lig_graph(dataset_df["smiles"][i]))
    for i in range(len(generate_df["smiles"])):
        generate_smiles.append(get_lig_graph(generate_df["smiles"][i]))


    dataset_dl  = DataLoader(dataset_smiles, batch_size=512, shuffle=False)
    generate_dl = DataLoader(generate_smiles, batch_size=512, shuffle=False)
    model = GATWithEdgeAttr(in_channels=9, out_channels=64, edge_attr_dim=3).cuda(7)

    dataset_embed  = []
    generate_embed = []
    for data in dataset_dl:
        data = data.cuda(7)
        out = model((data.x).float(), data.edge_index, data.edge_attr, data.batch)
        dataset_embed.append(out.detach().cpu())

    for data in generate_dl:
        data = data.cuda(7)
        out = model((data.x).float(), data.edge_index, data.edge_attr, data.batch)
        generate_embed.append(out.detach().cpu())

    dataset_embed = torch.concat(dataset_embed)
    generate_embed = torch.concat(generate_embed)
    cosine_similarity_matrix = cosine_similarity(dataset_embed, generate_embed)
    argmax_ind = np.argmax(cosine_similarity_matrix, axis = 1)
    xs = []
    ys = []
    zs = []
    gat_scores = []
    bas         = []
    for drug_id in range(len(dataset_df)):
        xs.append(generate_df.iloc[argmax_ind[drug_id]]['x'])
        ys.append(generate_df.iloc[argmax_ind[drug_id]]['y'])
        zs.append(generate_df.iloc[argmax_ind[drug_id]]['z'])
        bas.append(generate_df.iloc[argmax_ind[drug_id]]['ba'])
        gat_scores.append(cosine_similarity_matrix[drug_id, argmax_ind[drug_id]])
    dataset_df['x'] = xs
    dataset_df['y'] = ys
    dataset_df['z'] = zs
    dataset_df['ba'] = bas
    dataset_df['sim_score'] = gat_scores
    bas = np.array(bas)
    gat_scores = np.array(gat_scores)
    E_gat_scores_squared = math.sqrt(np.mean(gat_scores**2))
    E_bas_squared = math.sqrt(np.mean(bas**2))
    final_score = ((2/3) * gat_scores/E_gat_scores_squared) - ((1/3) * bas/(E_bas_squared))
    dataset_df['final_score'] = final_score
    sorted_df_multi = dataset_df.sort_values(by=['final_score'], ascending=[False])
    sorted_df_multi['index'] = np.arange(len(sorted_df_multi))
    gat_folder = f'drug_disease/{protein_name}/gat'
    if not os.path.exists(gat_folder):
        os.makedirs(gat_folder)
    print(gat_folder)
    sorted_df_multi.to_csv(os.path.join(gat_folder, 'gat.csv'))

drugbank_df = pd.read_csv('datasets/drugbank.csv')
paths = glob('drug_disease/*')
for path in paths:
    protein_name = os.path.split(path)[1]
    protein_path = f'drug_disease/{protein_name}/'
    equipath = os.path.join(protein_path, 'gat')
    generative_path = os.path.join(equipath, 'gat.csv')
    if not os.path.exists(equipath):
        continue
    
    shutil.copy(generative_path, f'gat/{protein_name}.csv')