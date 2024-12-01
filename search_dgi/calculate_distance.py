import pickle
from glob import glob
import pandas as pd
import os
import torch
import argparse



def create_potential_embed():
    potential_embed = sorted(glob("./potential_embed/*"), key = lambda x:int(os.path.split(x)[1].split('.')[0]))
    potential_tensor = []
    potential_index  = []
    for index, potential_path in enumerate(potential_embed):
        potential_tensor.append(torch.mean(pickle.load(open(potential_path, 'rb')), dim = 0))
        potential_index.append(int(os.path.split(potential_path)[1].split('.')[0]))

    potential_tensor = torch.stack(potential_tensor)
        
    filehandler = open(f"./matrix_dist/potential_tensor.pkl","wb") 
    pickle.dump(potential_tensor,filehandler)

    filehandler = open(f"./matrix_dist/potential_index.pkl","wb") 
    pickle.dump(potential_index,filehandler)


def create_chembl_embed(start, end):
    chembl_embed = sorted(glob("./chembl_embed/*"), key = lambda x:int(os.path.split(x)[1].split('.')[0]))
    chembl_tensor = []
    chembl_index  = []
    for index, chembl_path in enumerate(chembl_embed):
        if index < start:
            continue
        if index > end:
            break
        chembl_tensor.append(torch.mean(pickle.load(open(chembl_path, 'rb')), dim = 0))
        chembl_index.append(int(os.path.split(chembl_path)[1].split('.')[0]))


    chembl_tensor = torch.stack(chembl_tensor)
        
    filehandler = open(f"./matrix_dist/chembl_tensor_{start}_{end}.pkl","wb") 
    pickle.dump(chembl_tensor,filehandler)

    filehandler = open(f"./matrix_dist/chembl_index_{start}_{end}.pkl","wb") 
    pickle.dump(chembl_index,filehandler)
    
def potential_embed():
    # potential_ligand_data = pd.read_csv('./6dql.csv', delimiter= ",")
    # chembl_data           = pd.read_csv('../drugbank_downloader/chembl_34_chemreps.txt', delimiter= "\t")
    return

if __name__=="__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--start', type=int, help='start') 
    # parser.add_argument('--end', type=int, help='end') 
    # args = parser.parse_args()
    # create_potential_embed()
    # create_chembl_embed(int(args.start), int(args.end))
    
    potential_ligands_tensor  = pickle.load(open('matrix_dist/potential_tensor.pkl', 'rb')) 
    potential_ligands_index   = pickle.load(open('matrix_dist/potential_index.pkl', 'rb')) 
    potential_ligands_tensor  = potential_ligands_tensor.reshape(47,1,-1)
    for path in glob('matrix_dist/chembl_tensor_*'):
        start, end = os.path.split(path)[1].split('_')[2:]
        chembl_tensor  = pickle.load(open(f'matrix_dist/chembl_tensor_{start}_{end}', 'rb')) 
        chembl_index   = pickle.load(open(f'matrix_dist/chembl_index_{start}_{end}', 'rb'))
        chembl_tensor  = chembl_tensor.unsqueeze(0).repeat(47,1,1)
        print(potential_ligands_tensor.shape)
        print(chembl_tensor.shape)
        print(len(chembl_index))
        print(len(potential_ligands_index))
        print(torch.sum((potential_ligands_tensor - chembl_tensor)**2, dim = 2).shape)
        exit()
    # print(a.shape)