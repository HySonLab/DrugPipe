import numpy as np
import torch
from equiformer_pytorch import Equiformer
import pandas as pd
import pandas as pd
import numpy as np
from glob import glob
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
import shutil
import os
import torch
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from utils import *
from torch_geometric.nn import GCNConv
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import os
import pandas as pd
from torch.utils.data import DataLoader
import math
from sklearn.metrics.pairwise import cosine_similarity

class EquiformerDataloader(Dataset):
    def __init__(self, atoms_path = 'drugbank_atoms.pt', bonds_path = 'drugbank_bonds.pt', coords_path = 'drugbank_coords.pt', mask_path='drugbank_masks.pt'):
        self.drugbank_atoms  = torch.load(atoms_path)
        self.drugbank_bonds  = torch.load(bonds_path)
        self.drugbank_coords = torch.load(coords_path)
        self.drugbank_masks  = torch.load(mask_path)

    def __len__(self):
        return len(self.drugbank_atoms)

    def __getitem__(self, idx):
        return self.drugbank_atoms[idx].cuda(device_num), self.drugbank_bonds[idx].cuda(device_num), self.drugbank_coords[idx].cuda(device_num), self.drugbank_masks[idx].cuda(device_num)

df = pd.read_csv('datasets/drugbank.csv')
df['smiles'] = ' '
drugbank_atoms  = []
drugbank_bonds  = []
drugbank_coords = []
drugbank_masks  = []
for i in range(len(df)):
    row = df.iloc[i]
    drug_id = row['Drug id']
    supplier = Chem.SDMolSupplier(f'./datasets/drugbank_conformation/{drug_id}.sdf')
    one_conformer = supplier[1]
    atoms = torch.zeros(100)
    bonds = torch.zeros((100,100))
    n = one_conformer.GetNumAtoms()
    for index_atom, atom in enumerate(one_conformer.GetAtoms()):
        atoms[index_atom] = atom.GetAtomicNum()
    for bond in one_conformer.GetBonds():
        bond_type = -1
        str_bond_type = str(bond.GetBondType())
        begin_atom = bond.GetBeginAtomIdx()
        end_atom   = bond.GetEndAtomIdx()
        if (str_bond_type == 'SINGLE'):
            bond_type = 0
        elif (str_bond_type == 'DOUBLE'):
            bond_type = 1
        elif (str_bond_type == 'TRIPLE'):
            bond_type = 2
        elif (str_bond_type == 'AROMATIC'):
            bond_type = 3
        else:
            print(str_bond_type)
        bonds[begin_atom][end_atom] = bond_type
        bonds[end_atom][begin_atom] = bond_type
    coords = torch.zeros((100,3))
    coords[:n, :] =  torch.tensor(one_conformer.GetConformers()[0].GetPositions())
    mask = torch.concat((torch.ones(n), torch.zeros(100 - n))).bool()
    drugbank_atoms.append(atoms)
    drugbank_bonds.append(bonds)
    drugbank_coords.append(coords)
    drugbank_masks.append(mask)
device_num = 6
model = Equiformer(
    num_tokens = 100,
    dim = 64,
    num_edge_tokens = 4,       # number of edge type, say 4 bond types
    edge_dim = 16,             # dimension of edge embedding
    depth = 2,
    input_degrees = 1,
    num_degrees = 4,
    reduce_dim_out = True
).cuda(device_num)
dataloader = EquiformerDataloader()

train_dataloader = DataLoader(dataloader, batch_size=2, shuffle=False)
i = 0
lst = []
for sample in train_dataloader:
    out =  model(sample[0].int(), sample[2].float(), sample[3], edges = sample[1].int())
    i += 4
    lst.append(out.type0.detach().cpu())    

    torch.save(torch.cat(lst, dim = 0), 'drugbank_equiform.pt')
paths = glob('drug_disease/*')
for path in paths:
    protein_name = os.path.split(path)[1]
    protein_path = f'drug_disease/{protein_name}/'
    pdbqt_path   = f'drug_disease/{protein_name}/generation_docking/*.pdbqt'
    if not os.path.exists(os.path.join(protein_path, 'preprocessed_data')):
        continue
    protein_path = f'drug_disease/{protein_name}/'
    print(protein_path)
    pdbqt_path   = f'drug_disease/{protein_name}/generation_docking/*.pdbqt'
    generates = sorted(glob(pdbqt_path), key = lambda x: int(os.path.split(x)[1].split('.')[0]))
    generate_atoms  = []
    generate_bonds  = []
    generate_coords = []
    generate_masks  = []
    masks_valid     = []

    for generate in generates:
        os.system(f'obabel {generate} -O output.sdf')
        supplier = Chem.SDMolSupplier(f'output.sdf')
        one_conformer = supplier[0]
        atoms = torch.zeros(100)
        bonds = torch.zeros((100,100))
        try:
            n = one_conformer.GetNumAtoms()
            masks_valid.append(1)
            for index_atom, atom in enumerate(one_conformer.GetAtoms()):
                atoms[index_atom] = atom.GetAtomicNum()
        except:
            coords = torch.zeros((100,3))
            mask = torch.zeros(100).bool()
            masks_valid.append(0)
            generate_atoms.append(atoms)
            generate_bonds.append(bonds)
            generate_coords.append(coords)
            generate_masks.append(mask)
            continue
        for bond in one_conformer.GetBonds():
            bond_type = -1
            str_bond_type = str(bond.GetBondType())
            begin_atom = bond.GetBeginAtomIdx()
            end_atom   = bond.GetEndAtomIdx()
            if (str_bond_type == 'SINGLE'):
                bond_type = 0
            elif (str_bond_type == 'DOUBLE'):
                bond_type = 1
            elif (str_bond_type == 'TRIPLE'):
                bond_type = 2
            elif (str_bond_type == 'AROMATIC'):
                bond_type = 3
            else:
                print(str_bond_type)
            bonds[begin_atom][end_atom] = bond_type
            bonds[end_atom][begin_atom] = bond_type
        coords = torch.zeros((100,3))
        coords[:n, :] =  torch.tensor(one_conformer.GetConformers()[0].GetPositions())
        mask = torch.concat((torch.ones(n), torch.zeros(100 - n))).bool()
        generate_atoms.append(atoms)
        generate_bonds.append(bonds)
        generate_coords.append(coords)
        generate_masks.append(mask)

    equipath = os.path.join(protein_path, 'equiformer')
    if not os.path.exists(equipath):
        os.makedirs(equipath)
    torch.save(generate_atoms, os.path.join(equipath, 'equi_generate_atoms.pt'))
    torch.save(generate_bonds, os.path.join(equipath, 'equi_generate_bonds.pt'))
    torch.save(generate_coords, os.path.join(equipath, 'equi_generate_coords.pt'))
    torch.save(generate_masks, os.path.join(equipath, 'equi_generate_masks.pt'))
    torch.save(masks_valid, os.path.join(equipath, 'masks_valid.pt'))

paths = glob('drug_disease/*')
for path in paths:
    protein_name = os.path.split(path)[1]
    protein_path = f'drug_disease/{protein_name}/'
    equipath = os.path.join(protein_path, 'equiformer')
    if not os.path.exists(equipath):
        continue
    generate_atoms = torch.load(os.path.join(equipath, 'equi_generate_atoms.pt'))
    generate_bonds = torch.load(os.path.join(equipath, 'equi_generate_bonds.pt'))
    generate_coords = torch.load(os.path.join(equipath, 'equi_generate_coords.pt'))
    generate_masks = torch.load(os.path.join(equipath, 'equi_generate_masks.pt'))
    masks_valid = torch.load(os.path.join(equipath, 'masks_valid.pt'))

    dataloader = EquiformerDataloader(os.path.join(equipath, 'equi_generate_atoms.pt'), os.path.join(equipath, 'equi_generate_bonds.pt'),
                                      os.path.join(equipath, 'equi_generate_coords.pt'), os.path.join(equipath, 'equi_generate_masks.pt'))
    train_dataloader = DataLoader(dataloader, batch_size=2, shuffle=False)

    generative_embed = []
    for sample in train_dataloader:
        out = model(sample[0].int(), sample[2].float(), sample[3], edges = sample[1].int())
        generative_embed.append(out.type0.detach().cpu())
    torch.save(torch.concat(generative_embed, dim = 0), os.path.join(equipath, 'generative_embed.pt'))
drugbank_equiform = torch.load('drugbank_equiform.pt')
drugbank_df = pd.read_csv('datasets/drugbank.csv')
paths = glob('drug_disease/*')
for path in paths:
    protein_name = os.path.split(path)[1]
    protein_path = f'drug_disease/{protein_name}/'
    equipath = os.path.join(protein_path, 'equiformer')
    generative_path = os.path.join(equipath, 'generative_embed.pt')
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
    sorted_df_multi.to_csv(equipath + '/equiscore.csv')
    print(equipath)


from sklearn.metrics.pairwise import cosine_similarity
drugbank_equiform = torch.load('drugbank_equiform.pt')
drugbank_df = pd.read_csv('datasets/drugbank.csv')
paths = glob('drug_disease/*')
for path in paths:
    protein_name = os.path.split(path)[1]
    protein_name = os.path.split(path)[1]
    protein_path = f'drug_disease/{protein_name}/'
    equipath = os.path.join(protein_path, 'equiformer')
    generative_path = os.path.join(equipath, 'generative_embed.pt')
    if not os.path.exists(generative_path):
        continue
    shutil.copy(equipath + '/equiscore.csv', f'equiform/{protein_name}.csv')