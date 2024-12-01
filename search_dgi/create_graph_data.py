import pandas as pd
from rdkit import Chem
import networkx as nx
from torch_geometric.utils.convert import to_networkx, from_networkx
from torch_geometric.loader import DataLoader
import pickle
from ogb.utils.mol import smiles2graph
from torch_geometric.data import Batch, Data
import torch
import os

def get_lig_graph(smiles):
        graph = smiles2graph(smiles)
        return Data(x = torch.tensor(graph['node_feat']), edge_index = torch.tensor(graph['edge_index']), edge_attr = torch.tensor(graph['edge_feat']))

def create_drugbank_graph_data(path_to_dataset = "./",name_dataset = 'drugbank_approved'):
    drug_bank     = pd.read_csv(os.path.join(path_to_dataset,f'{name_dataset}.csv'), delimiter= ",")
    data = []
    wrong_smile = []
    right_smile = []
    for i in range(len(drug_bank['smiles'])):
        try:
            
            data.append(get_lig_graph(drug_bank['smiles'][i]))
            right_smile.append(i)
        except:
            wrong_smile.append(i)
    graph_data_path = "/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/graph_data"
    # filehandler = open(f"graph_data_drugbank.pkl","wb") 
    filehandler = open(os.path.join(graph_data_path, f"{name_dataset}_embed.pkl"),"wb")
    pickle.dump(data,filehandler)
    filehandler.close()
    filehandler = open(os.path.join(graph_data_path, f"wrong_smiles_{name_dataset}.pkl"),"wb") 
    pickle.dump(wrong_smile,filehandler)
    filehandler.close()
    filehandler = open(os.path.join(graph_data_path,f"right_smiles_{name_dataset}.pkl"),"wb") 
    pickle.dump(right_smile,filehandler)
    filehandler.close()
    
path_dataset = "/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/data"
create_drugbank_graph_data(path_to_dataset = path_dataset)

# with open('/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/graph_data/drugbank_approved_embed.pkl', 'rb') as file:
#     my_object = pickle.load(file)
# print(len(my_object))

# with open('/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/graph_data/right_smiles_drugbank_approved.pkl', 'rb') as file:
#     my_object = pickle.load(file)
# print(len(my_object))

# with open('/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/graph_data/wrong_smiles_drugbank_approved.pkl', 'rb') as file:
#     my_object = pickle.load(file)
# print(len(my_object))