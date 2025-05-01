import sys
sys.path.append('/home/phuc.phamhuythienai@gmail.com/Desktop/New/DrugPipe') 
import pandas as pd
from rdkit import Chem
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from rdkit.DataStructs import TanimotoSimilarity
import os
import sys
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from egnn_pytorch import EGNN_Network
from equiformer_pytorch import Equiformer
from itertools import combinations
import numpy as np
from rdkit import Chem
from rdkit.Chem import BRICS
from collections import Counter
# Thêm vào cuối danh sách sys.path

path_dti = '/home/phuc.phamhuythienai@gmail.com/Desktop/New/DrugPipe/datasets/protein_disease/dti_dataset.csv' 
path_drugbank = '/home/phuc.phamhuythienai@gmail.com/Desktop/New/DrugPipe/datasets/drugbank.csv'
drugbank_data = pd.read_csv(path_drugbank)
df = pd.read_csv(path_dti)
def create_data(path_dti = path_dti , path_drugbank = path_drugbank):
    df = pd.read_csv(path_dti)
    drugbank_data = pd.read_csv(path_drugbank)
    retain_list = list(drugbank_data['Drug id'])
    for i in range(len(df)):
        lst = [drug for drug in df.loc[i]['Drug IDs'].split(';') if drug.strip() in retain_list]
        if len(lst) < 2:
            df.at[i, 'Drug IDs'] = 'NA'
        else:
            df.at[i, 'Drug IDs'] = ";".join(lst)
    df = df[df['Drug IDs'] != 'NA']
    df.to_csv('multi_drugs_target.csv')

def morgan_fingerprint_sim(path_drugbank = path_drugbank):
    drugbank_data = pd.read_csv(path_drugbank)
    # Convert SMILES to RDKit molecules
    drugbank_data["Molecule"] = drugbank_data["smiles"].apply(Chem.MolFromSmiles)
    # Compute Morgan fingerprints (radius=2, 2048-bit vector)
    drugbank_data["Morgan_Fingerprint"] = drugbank_data["Molecule"].apply(lambda mol: AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
    # Compute pairwise Tanimoto similarities
    num_mols = len(drugbank_data)
    sim_matrix = np.zeros((num_mols , num_mols))
    for i in range(num_mols):
        for j in range(i, num_mols):
            print(i,j)
            sim = TanimotoSimilarity(drugbank_data["Morgan_Fingerprint"].iloc[i], drugbank_data["Morgan_Fingerprint"].iloc[j])
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim  # Since the matrix is symmetric
    np.save('morgan_fingerprint_sim_matrix.npy', sim_matrix)

def tanimoto_sim(path_drugbank = path_drugbank):
    drugbank_data = pd.read_csv(path_drugbank)
    num_mols = len(drugbank_data)
    sim_matrix = np.zeros((num_mols , num_mols))
    for i in range(num_mols):
        for j in range(i, num_mols):
            print(i,j)
            try:
                sim = similarity(drugbank_data["smiles"][i], drugbank_data["smiles"][j])
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim  # Since the matrix is symmetric
            except:
                continue
    np.save('tanimoto_sim_matrix.npy', sim_matrix)

def gnn_sim(path_drugbank = path_drugbank, device = 'cuda:0'):
    gconv = GConv(
        input_dim=9, hidden_dim=64, activation=torch.nn.ReLU, num_layers=3
    ).to(device)
    fc1 = FC(hidden_dim=64 * 3)
    fc2 = FC(hidden_dim=64 * 3)
    encoder_model = Encoder(encoder=gconv, local_fc=fc1, global_fc=fc2).to(device)
    encoder_model.eval()
    
    
    drugbank_data = pd.read_csv(path_drugbank)
    smile_drugbank = [] 
    for i in range(len(drugbank_data["smiles"])):
        smile_drugbank.append(get_lig_graph(drugbank_data["smiles"][i]))


    dataloader = DataLoader(smile_drugbank, batch_size=512, shuffle=False)
    smile_drugbank_embed_list = embed_data(encoder_model, dataloader, device)

    docking_generation_embed = torch.stack(smile_drugbank_embed_list).cpu().detach().numpy()
    cosine_sim_matrix = cosine_similarity(docking_generation_embed)
    print(cosine_sim_matrix.shape)
    np.save('gnns_sim_matrix.npy', cosine_sim_matrix)

def gat_sim(path_drugbank = path_drugbank, device = 'cuda:0'):
    
    drugbank_data = pd.read_csv(path_drugbank)
    smile_drugbank = [] 
    for i in range(len(drugbank_data["smiles"])):
        smile_drugbank.append(get_lig_graph(drugbank_data["smiles"][i]))


    dataloader = DataLoader(smile_drugbank, batch_size=512, shuffle=False)
    model = GATWithEdgeAttr(in_channels=9, out_channels=64, edge_attr_dim=3).to(device)
    
    dataset_embed  = []
    for data in dataloader:
        data = data.to(device)
        out = model((data.x).float(), data.edge_index, data.edge_attr, data.batch)
        dataset_embed.append(out.detach().cpu())
    
    dataset_embed = torch.concat(dataset_embed)
    cosine_similarity_matrix = cosine_similarity(dataset_embed)
    print(cosine_similarity_matrix.shape)
    np.save('gats_sim_matrix.npy', cosine_similarity_matrix)

def create_threed_data(path_drugbank = path_drugbank):
    df = pd.read_csv(path_drugbank)
    df['smiles'] = ' '
    drugbank_atoms  = []
    drugbank_bonds  = []
    drugbank_coords = []
    drugbank_masks  = []
    for i in range(len(df)):
        row = df.iloc[i]
        drug_id = row['Drug id']
        n = 0
        atoms = torch.zeros(100)
        bonds = torch.zeros((100,100))
        coords = torch.zeros((100,3))
        mask = torch.concat((torch.ones(n), torch.zeros(100 - n))).bool()
        try:
            supplier = Chem.SDMolSupplier(f'/home/phuc.phamhuythienai@gmail.com/Desktop/New/DrugPipe/datasets/drugbank_conformation/{drug_id}.sdf')
        except:
            drugbank_atoms.append(atoms)
            drugbank_bonds.append(bonds)
            drugbank_coords.append(coords)
            drugbank_masks.append(mask)
            continue

        
        if len(supplier) == 1:
            one_conformer = supplier[0]
            
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
            coords[:n, :] =  torch.tensor(one_conformer.GetConformers()[0].GetPositions())
            mask = torch.concat((torch.ones(n), torch.zeros(100 - n))).bool()
        drugbank_atoms.append(atoms)
        drugbank_bonds.append(bonds)
        drugbank_coords.append(coords)
        drugbank_masks.append(mask)

    torch.save(drugbank_atoms,'drugbank_atoms.pt')
    torch.save(drugbank_bonds,'drugbank_bonds.pt')
    torch.save(drugbank_coords,'drugbank_coords.pt')
    torch.save(drugbank_masks,'drugbank_masks.pt')


class E3GNNDataloader(Dataset):
    def __init__(self, atoms_path = 'drugbank_atoms.pt', bonds_path = 'drugbank_bonds.pt', coords_path = 'drugbank_coords.pt', mask_path='drugbank_masks.pt', device_num = 0):
        self.drugbank_atoms  = torch.load(atoms_path)
        self.drugbank_bonds  = torch.load(bonds_path)
        self.drugbank_coords = torch.load(coords_path)
        self.drugbank_masks  = torch.load(mask_path)
        self.device          = device_num

    def __len__(self):
        return len(self.drugbank_atoms)

    def __getitem__(self, idx):
        return self.drugbank_atoms[idx].cuda(self.device), self.drugbank_bonds[idx].cuda(self.device), self.drugbank_coords[idx].cuda(self.device), self.drugbank_masks[idx].cuda(self.device)
    

class EquiformerDataloader(Dataset):
    def __init__(self, atoms_path = 'drugbank_atoms.pt', bonds_path = 'drugbank_bonds.pt', coords_path = 'drugbank_coords.pt', mask_path='drugbank_masks.pt', device_num = 0):
        self.drugbank_atoms  = torch.load(atoms_path)
        self.drugbank_bonds  = torch.load(bonds_path)
        self.drugbank_coords = torch.load(coords_path)
        self.drugbank_masks  = torch.load(mask_path)
        self.device          = device_num

    def __len__(self):
        return len(self.drugbank_atoms)

    def __getitem__(self, idx):
        return self.drugbank_atoms[idx].cuda(self.device ), self.drugbank_bonds[idx].cuda(self.device ), self.drugbank_coords[idx].cuda(self.device ), self.drugbank_masks[idx].cuda(self.device )


def egnn_sim():
    net = EGNN_Network(
        num_tokens = 100,
        dim = 64,
        depth = 3,
        only_sparse_neighbors = True
    ).cuda('cuda:0')
    dataloader = E3GNNDataloader()
    train_dataloader = DataLoader(dataloader, batch_size=256, shuffle=False)
    generate_lst = []
    for sample in train_dataloader:
        feats_out, coords_out = net(sample[0].int(), sample[2].float(), mask = sample[3].bool(), adj_mat = sample[1].bool())
        generate_lst.append(torch.mean(feats_out, dim = 2).detach().cpu())

    generate_embed = torch.concatenate(generate_lst)

    cosine_similarity_matrix = cosine_similarity(generate_embed)
    np.save('egnn_sim_matrix.npy', cosine_similarity_matrix)
    
def equiformer_sim():
    model = Equiformer(
    num_tokens = 100,
    dim = 64,
    num_edge_tokens = 4,       # number of edge type, say 4 bond types
    edge_dim = 16,             # dimension of edge embedding
    depth = 2,
    input_degrees = 1,
    num_degrees = 4,
    reduce_dim_out = True
    ).cuda('cuda:0')
    dataloader = EquiformerDataloader()
    train_dataloader = DataLoader(dataloader, batch_size=2, shuffle=False)
    lst = []
    for sample in train_dataloader:
        out =  model(sample[0].int(), sample[2].float(), sample[3], edges = sample[1].int())
        lst.append(out.type0.detach().cpu())    

    torch.save(torch.cat(lst, dim = 0), 'drugbank_equiform.pt')
    cosine_similarity_matrix = cosine_similarity(generate_embed)
    np.save('equiformer_sim_matrix.npy', cosine_similarity_matrix)

def write_txt_matching_real_drug(name_lst = ['tanimoto', 'morgan', 'gnns', 'gats', 'egnn' , 'equiformer'],path_drugbank = path_drugbank):
    name_lst = ['tanimoto', 'morgan', 'gnns', 'gats', 'egnn' , 'equiformer']
    for name in name_lst:
        matrix = np.load(f'{name}_sim_matrix.npy')
        dict_representation = drugbank_data.set_index("Drug id")["Key idx"].to_dict()
        with open(f'text/similarity_ranks_{name}.txt', "w") as f:
            for i in range(len(df)):
                drug_ids = [dict_representation[drug.strip()] for drug in df.iloc[i]['Drug IDs'].split('; ')]
                for idx in drug_ids:
                    for idxb in drug_ids:
                        similarities = matrix[idx]  # Get similarity row for this drug
                        sorted_similarities = np.argsort(-similarities)  # Sort descending
                        rank = np.where(sorted_similarities == idxb)[0][0] + 1  # Find self-index and convert to rank (1-based)
                        if rank == 1:
                            continue
                        f.write(f"Drug at index {idxb} (ID: {drugbank_data['Drug id'][idxb]}) has similarity rank: {rank}\n")
                    f.write("======================\n")

def brics_similarity_analysis(path_drugbank = path_drugbank):
    drugbank_data = pd.read_csv(path_drugbank)
    # Define total dataset size
    total_drugs = 9716  
    name_lst = ['tanimoto', 'morgan', 'gnns', 'gats', 'egnn' , 'equiformer']
    # Dictionaries to store results for each method
    all_groups_dict = {}  # Stores rank groups
    all_index_dict = {}   # Stores index groups
    for name in name_lst:
        # Read ranks from file
        file_path = f'text/similarity_ranks_{name}.txt'  # Update this with the actual file path
        # Reload the similarity ranks and extract multiple groups
        all_groups    = []
        current_group = []
        all_index     = []
        current_index = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if "======" in line:  # Detect new group
                    if current_group:  # Save previous group if not empty
                        all_groups.append(current_group)
                        all_index.append(current_index)
                    current_group = []  # Start a new group
                    current_index = []
                elif "has similarity rank" in line:
                    rank = int(line.split("rank:")[-1].strip())
                    index = line.split("ID:")[-1][:8].strip()
                    current_group.append(rank)
                    current_index.append(index)
                    

        # Add the last group if it wasn't added
        if current_group:
            all_groups.append(current_group)
        if current_index:
            all_index.append(current_index)     
        all_groups_dict[name] = all_groups
        all_index_dict[name] = all_index
    total_brics_dict = {}
    for i in range(len(all_groups_dict[name])):
        current_dict_index     = {}
        current_dict_drug_name = {}
        drug_ids_dict = {}
        for name in name_lst:
            current_dict_index[name] = all_groups_dict[name][i]
            current_dict_drug_name[name] = all_index_dict[name][i]
            drug_ids_dict[name] = all_index_dict[name][i]
        res = {}    
        for key, values in current_dict_index.items():
            sorted_indices = sorted(range(len(values)), key=lambda i: values[i])
            res[key] = [current_dict_drug_name[key][sorted_indices[0]], current_dict_drug_name[key][sorted_indices[1]]]# Lấy index của hai giá trị nhỏ nhất
        for name in name_lst:
            drug_ids = res[name]
            # Tạo danh sách SMILES từ DrugBank
            smiles_string = [drugbank_data.loc[drugbank_data['Drug id'] == drug_id.strip(), 'smiles'].values[0] for drug_id in drug_ids]

            from itertools import combinations

            # Lưu BRICS fragments cho mỗi thuốc
            drug_brics_dict = {}

            for drug_id, smiles in zip(drug_ids, smiles_string):
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    fragments = set(BRICS.BRICSDecompose(mol))  # Dùng set để so sánh nhanh hơn
                    drug_brics_dict[drug_id] = fragments

            # So sánh từng cặp thuốc
            common_brics_count = {}

            for (drug1, drug2) in combinations(drug_ids, 2):  
                common_fragments = drug_brics_dict[drug1] & drug_brics_dict[drug2]  # Lấy giao nhau giữa 2 tập
                common_brics_count[(drug1, drug2)] = len(common_fragments)
                union_set = drug_brics_dict[drug1].union(drug_brics_dict[drug2])
            # In kết quả
            print(f"Cặp thuốc {name}| Số mảnh BRICS trùng nhau")
            for pair, count in common_brics_count.items():
                print(f"{pair} | {count}")
            if not total_brics_dict.get(name):
                total_brics_dict[name] = 0
            total_brics_dict[name] +=  len(common_fragments)/len(union_set)
            
        
write_txt_matching_real_drug()