import os
import sys
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from Bio import PDB
from tdc.chem_utils.oracle.oracle import similarity
import pickle
from torch_geometric.data import Data
import torch
from ogb.utils.mol import smiles2graph
from search_dgi.info_graph import *
from sklearn.metrics.pairwise import cosine_similarity
from paretoset import paretoset
from tdc.chem_utils.oracle.oracle import Vina_3d
import atom3d.util.formats as fo
import LigPrepper
import mlcrate as mlc
from glob import glob
import scipy
from sklearn.metrics.pairwise import euclidean_distances
import warnings
import esm
import lmdb
from transformers import EsmModel, EsmTokenizer
from Bio.PDB import PDBParser
from torch_geometric.data import HeteroData
from torchdrug import data as td
from rdkit.Chem import AllChem
from tqdm import tqdm
import argparse
import random
# Defining main function 
warnings.filterwarnings("ignore")


def Seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
def create_ligand(smiles, file_path):
    mol = Chem.MolFromSmiles(smiles)

    # Add hydrogen atoms
    mol = Chem.AddHs(mol)

    # Generate 3D coordinates
    Chem.AllChem.EmbedMolecule(mol, Chem.AllChem.ETKDG())

    # Write the molecule to an SDF file
    writer = Chem.SDWriter(file_path)
    writer.write(mol)
    writer.close()

three_to_one = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}

def create_ligand(smiles, file_path):
    mol = Chem.MolFromSmiles(smiles)

    # Add hydrogen atoms
    mol = Chem.AddHs(mol)

    # Generate 3D coordinates
    Chem.AllChem.EmbedMolecule(mol, Chem.AllChem.ETKDG())

    # Write the molecule to an SDF file
    writer = Chem.SDWriter(file_path)
    writer.write(mol)
    writer.close()

# Helper function
def load_pkl(name):
    return pickle.load(open(f"{name}", "rb"))

# create folder and merge path to the folder
def create_and_merge_path(out_folder, file_name):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    out_file = os.path.join(out_folder, file_name)
    return out_file

def read_csv_idx(csv_path):
    data = pd.read_csv(csv_path)
    data.rename(columns={"Unnamed: 0": "key_idx"}, inplace=True)
    return data

# ========= VINA DOCKING EXPERIMENT =========
# 1. Remove water before docking
def remove_water(input_pdb_file, output_pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("structure", input_pdb_file)

    # Create a new structure without water molecules
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(output_pdb_file, select=SelectProtein())

class SelectProtein(PDB.Select):
    def accept_atom(self, atom):
        # Select atoms that belong to residues which are not water
        return atom.get_parent().get_resname() not in ["HOH", "WAT", "SOL"]

## Using: remove_water(input_pdb_file, output_pdb_file)

# 2. Convert protein from pdb to pdbqt file
def convert_pdb_to_pdbqt(pdb_file, pdbqt_file):
    cmd = f"obabel {pdb_file} -O {pdbqt_file} -h -gen3D -xr"
    # Using os.system() method
    os.system(cmd)

## Using: convert_pdb_to_pdbqt(pdb_file, pdbqt_file)

def prepare_protein_pdbqt(pdb_path,path_pdbqt):
    # 1. Remove water
    output_pdb_file = pdb_path.replace('.pdb', '') + "_remove_water.pdb"
    remove_water(pdb_path, output_pdb_file)
    convert_pdb_to_pdbqt(output_pdb_file, path_pdbqt)
    os.remove(output_pdb_file)

# ========= SIMILARITY MATCHING =========
# a) Tanimoto distance score
def tanimoto_score(path_protein, path_dataset):
    generate = pd.read_csv(path_protein, encoding="utf-8")
    drug_bank = pd.read_csv(path_dataset, delimiter=",")
    output = []
    for i in range(len(drug_bank["smiles"])):
        max_sim = 0
        min_ba = 0
        for j in range(len(generate["smiles"])):
            try:
                tanimoto = similarity(drug_bank["smiles"][i], generate["smiles"][j])
                if tanimoto > max_sim:
                    max_sim = tanimoto
                    min_ba = generate["BA"][j]
            except:
                continue
        output.append([max_sim, min_ba])
    output = np.array(output)
    out_folder = "/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/similarity_score"
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    out_file = os.path.join(out_folder, f"tanimoto_{name_protein}_{name_dataset}.pkl")
    filehandler = open(out_file, "wb")
    pickle.dump(output, filehandler)

# b) Morgan distance score
def morgan_score(name_protein="5dl2", name_dataset="drugbank"):
    generate = pd.read_csv(f"data/{name_protein}.csv", encoding="utf-8")
    drug_bank = pd.read_csv(f"data/{name_dataset}.csv", delimiter=",")
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2)
    output = []
    for i in range(len(drug_bank["smiles"])):
        max_sim = 0
        min_ba = 0
        for j in range(len(generate["smiles"])):
            try:
                m1 = Chem.MolFromSmiles(drug_bank["smiles"][i])
                m2 = Chem.MolFromSmiles(generate["smiles"][j])
                fp1 = fpgen.GetSparseCountFingerprint(m1)
                fp2 = fpgen.GetSparseCountFingerprint(m2)
                morgan_sim = DataStructs.DiceSimilarity(fp1, fp2)
                if morgan_sim > max_sim:
                    max_sim = morgan_sim
                    min_ba = generate["BA"][j]
            except:
                continue
        output.append([max_sim, min_ba])
    output = np.array(output)
    out_folder = "/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/similarity_score"
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    out_file = os.path.join(out_folder, f"morgan_{name_protein}_{name_dataset}.pkl")
    filehandler = open(out_file, "wb")
    pickle.dump(output, filehandler)

# c) GNNs distance score
# 1. Convert compounds to graph data
def get_lig_graph(smiles):
    graph = smiles2graph(smiles)
    return Data(
        x=torch.tensor(graph["node_feat"]),
        edge_index=torch.tensor(graph["edge_index"]),
        edge_attr=torch.tensor(graph["edge_feat"]),
    )


def create_graph_data(name_file="6dql"):
    drug_bank = pd.read_csv(f"data/{name_file}.csv", delimiter=",")
    data = []
    wrong_smile = []
    right_smile = []
    for i in range(len(drug_bank["smiles"])):
        try:

            data.append(get_lig_graph(drug_bank["smiles"][i]))
            right_smile.append(i)
        except:
            wrong_smile.append(i)
    output_files = "graph_data"
    if not os.path.exists(output_files):
        os.makedirs(output_files)
    graph_out_path = os.path.join("graph_data", f"graph_{name_file}.pkl")
    filehandler = open(graph_out_path, "wb")
    pickle.dump(data, filehandler)
    filehandler.close()
    wrong_smiles_out_path = os.path.join("graph_data", f"wrong_smiles_{name_file}.pkl")
    filehandler = open(wrong_smiles_out_path, "wb")
    pickle.dump(wrong_smile, filehandler)
    filehandler.close()
    right_smiles_out_path = os.path.join("graph_data", f"right_smiles_{name_file}.pkl")
    filehandler = open(right_smiles_out_path, "wb")
    pickle.dump(right_smile, filehandler)
    filehandler.close()


# 2. Embed graph to vector
def embed_data(encoder_model, dataloader, device):
    encoder_model.eval()
    embed_lst = []
    for data in dataloader:
        data = data.to(device)
        z, g = encoder_model(data.x, data.edge_index, data.batch)
        z, g = encoder_model.project(z, g)
        for i in torch.unique(data.batch):
            embed_lst.append(g[i])
    return embed_lst

def create_embedding(name_file="6dql", device = "cuda:5"):
    dataset = pickle.load(open(f"/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/graph_data/graph_{name_file}.pkl", "rb"))
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False)
    gconv = GConv(
        input_dim=9, hidden_dim=64, activation=torch.nn.ReLU, num_layers=3
    ).to(device)
    fc1 = FC(hidden_dim=64 * 3)
    fc2 = FC(hidden_dim=64 * 3)
    encoder_model = Encoder(encoder=gconv, local_fc=fc1, global_fc=fc2).to(device)
    encoder_model.load_state_dict(torch.load("/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/search_dgi/encoder_best.pt"))
    encoder_model.eval()
    embed_lst = embed_data(encoder_model, dataloader)
    filehandler = open(f"/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/graph_data/{name_file}_embed.pkl", "wb")
    pickle.dump(embed_lst, filehandler)
    filehandler.close()

def gnns_score(protein="6dql", dataset="drugbank"):
    drug_bank_embed = torch.stack(
        pickle.load(open(f"/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/graph_data/{dataset}_embed.pkl", "rb"))
    )
    embedding_dim = drug_bank_embed.shape[1]
    potential_embed = torch.stack(
        pickle.load(open(f"/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/graph_data/{protein}_embed.pkl", "rb"))
    )
    num_potential = potential_embed.shape[0]
    wrong_smiles = pickle.load(open(f"/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/graph_data/wrong_smiles_{dataset}.pkl", "rb"))
    right_smiles = pickle.load(open(f"/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/graph_data/right_smiles_{dataset}.pkl", "rb"))
    drug_bank_mask_error = torch.zeros(
        (len(wrong_smiles) + len(right_smiles), embedding_dim)
    ).cuda()
    drug_bank_mask_error[right_smiles] = drug_bank_embed
    potential_embed = potential_embed.detach().cpu().numpy()
    drug_bank_mask_error = drug_bank_mask_error.detach().cpu().numpy()
    gnns_sim = cosine_similarity(drug_bank_mask_error, potential_embed)
    infinity_embed = np.ones((len(wrong_smiles), num_potential)) * -1
    gnns_sim[wrong_smiles] = infinity_embed
    generate = pd.read_csv(f"data/{protein}.csv", encoding="utf-8")
    potential_ligand_ba = list(generate["BA"])
    argmax_index = np.argmax(gnns_sim, axis=1)
    output = np.hstack(
        [
            gnns_sim[np.arange(0, argmax_index.shape[0]), argmax_index].reshape(-1, 1),
            np.array(potential_ligand_ba)[argmax_index].reshape(-1, 1),
        ]
    )
    out_folder = "/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/similarity_score"
    file_name = f"gnns_{protein}_{dataset}.pkl"
    out_file = create_and_merge_path(out_folder, file_name)
    filehandler = open(out_file, "wb")
    pickle.dump(output, filehandler)
    filehandler.close()

# ========= Save preprocessed data =========
def save_preprocessed_data(name_protein="6dql", dataset="drugbank"):
    tanimoto_arr = load_pkl(f"/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/similarity_score/tanimoto_{name_protein}_{dataset}.pkl")
    morgan_arr = load_pkl(f"/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/similarity_score/morgan_{name_protein}_{dataset}.pkl")
    gnns_arr = load_pkl(f"/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/similarity_score/gnns_{name_protein}_{dataset}.pkl")
    path = f"/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/data/{dataset}.csv"
    database = pd.read_csv(f"/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/data/{dataset}.csv")
    database["tanimoto_score"] = tanimoto_arr[:, 0]
    database["tanimoto_ba"] = tanimoto_arr[:, 1]
    database["morgan_score"] = morgan_arr[:, 0]
    database["morgan_ba"] = morgan_arr[:, 1]
    database["gnns_score"] = gnns_arr[:, 0]
    database["gnns_ba"] = gnns_arr[:, 1]
    out_folder = "/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/preprocessed_data"
    file_name = f"preprocessed_{name_protein}_{dataset}_data.csv"
    out_file = create_and_merge_path(out_folder, file_name)
    database.to_csv(out_file, index=False)

# ========= Multi-objective ranking stage 1 ==========
def multi_objective_filter_orange(protein_name="6dql", dataset="drugbank"):
    preprocessed_data = read_csv_idx(
        f"/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/preprocessed_data/preprocessed_{protein_name}_{dataset}_data.csv"
    )
    all_score = preprocessed_data[
        [
            "tanimoto_score",
            "tanimoto_ba",
            "morgan_score",
            "morgan_ba",
            "gnns_score",
            "gnns_ba",
        ]
    ]
    mask_all = paretoset(all_score, sense=["max", "min", "max", "min", "max", "min"])
    paretoset_all = preprocessed_data[mask_all]
    return paretoset_all

# ========= Docking 1 - Create small dataset for finetune process ========
def docking_1(protein_name="6dql", dataset="drugbank"):
    filtered_ligands = multi_objective_filter_orange(protein_name, dataset)

    receptor_file_path = f"/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/protein_docking/{protein_name}.pdbqt"
    protein_df = fo.bp_to_df(fo.read_pdb(receptor_file_path))
    center_x, center_y, center_z = (
        np.mean(protein_df["x"]),
        np.mean(protein_df["y"]),
        np.mean(protein_df["z"]),
    )
    func = Vina_3d(
        receptor_file_path,
        [float(center_x), float(center_y), float(center_z)],
        [35.0, 35.0, 35.0],
    )
    ba_lst = np.ones(len(filtered_ligands["smiles"]))
    for idx, smiles in enumerate(filtered_ligands["smiles"]):
        try:
            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), True)
            LigPrepper.smiles2pdbqt(smiles, labels=f"generate_{protein_name}")
            out_folder = f"/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/docking_output/{protein_name}_{dataset}_binding_ligands"
            name_out_ligands = str(filtered_ligands["key_idx"].iloc[idx])
            out_ligand = create_and_merge_path(
                out_folder, f"{str(name_out_ligands)}.pdbqt"
            )
            ba_generation = func(f"generate_{protein_name}.pdbqt", out_ligand)
            ba_lst[idx] = ba_generation
        except:
            ba_lst[idx] = 0.0
        filtered_ligands["BA"] = ba_lst
        filtered_ligands.to_csv(
            f"/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/docking_output/{protein_name}_{dataset}_docking1.csv", index=False
        )
        os.remove(f"generate_{protein_name}.pdbqt")

# ========== Dataset creation ============
# Function to calculate distance between two atoms
def extract_ith_list(lists_of_lists, idx):
    return [sublist[idx] for sublist in lists_of_lists]

def convert_pdbqt_to_pdb(protein_name="5dl2", database="drugbank", stage = 2):
    ligand_docking_out = glob(
        f"/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/docking_output/{protein_name}_{database}_binding_ligands_stage_{stage}/*.pdbqt"
    )
    for ligand_path in ligand_docking_out:
        file_name = os.path.split(ligand_path)[1].split(".")[0] + ".pdb"
        out_folder = os.path.join(
            f"/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/ligands_conformation/stage_{stage}", f"{protein_name}_{database}_binding_ligands"
        )
        outfile = create_and_merge_path(out_folder, file_name)
        print(outfile)
        cmd = f"obabel {ligand_path} -O {outfile}"
        os.system(cmd)
    protein_path = f"/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/protein_docking/{protein_name}.pdbqt"
    out_protein = f"/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/ligands_conformation/stage_{stage}/{protein_name}.pdb"
    cmd = f"obabel {protein_path} -O {out_protein}"
    os.system(cmd)

def pocket_analysis(protein_name="6dql", dataset="drugbank", num_pose=10, cutoff=5, stage = 2):
    protein_path = f"/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/ligands_conformation/stage_{stage}/{protein_name}.pdb"
    ligands_folder = f"/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/ligands_conformation/stage_{stage}/{protein_name}_{dataset}_binding_ligands/"
    protein_pdb = Chem.MolFromPDBFile(protein_path, removeHs=False)
    protein_residue_list = []
    protein_coord_list = []
    for atom in protein_pdb.GetAtoms():
        residue_info = atom.GetPDBResidueInfo()
        if residue_info:
            atom_idx = atom.GetIdx()
            res_name = residue_info.GetResidueName().strip()
            res_num = residue_info.GetResidueNumber()
            chain_id = residue_info.GetChainId()
            pos = protein_pdb.GetConformer().GetAtomPosition(atom_idx)
            protein_residue_list.append(f"{chain_id}-{res_name}.{res_num}")
            protein_coord_list.append([pos.x, pos.y, pos.z])

    protein_coord = np.array(protein_coord_list)
    protein_residue = np.array(protein_residue_list)

    ligand_paths = glob(ligands_folder + "/*")
    ligand_lst = []
    pocket_lst = []
    for ligand_path in ligand_paths:
        ligand_lst.append(int(os.path.split(ligand_path)[1].split(".")[0]))
        try:
            ligand_pdb = Chem.MolFromPDBFile(ligand_path, removeHs=False)
            lst = []
            number_of_conformer = ligand_pdb.GetNumConformers()
            for conformer_id in range(number_of_conformer):
                ligand_coords = ligand_pdb.GetConformer(conformer_id).GetPositions()
                pairwise_distance = euclidean_distances(ligand_coords, protein_coord)
                less_than_cutoff_mask = np.where(pairwise_distance < cutoff)
                cutoff_res = list(
                    set(protein_residue[list(set(less_than_cutoff_mask[1]))])
                )
                lst.append(cutoff_res)
            if len(lst) < num_pose:
                for _ in range(num_pose - len(lst)):
                    lst.append([])
            pocket_lst.append(lst)
        except:
            empty = [[] for _ in range(num_pose)]
            pocket_lst.append(empty)
    protein_processed_data = pd.read_csv(
        f"/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/docking_output/{protein_name}_{dataset}_docking{stage}.csv"
    )
    filtered_df = protein_processed_data[
        protein_processed_data["key_idx"].isin(ligand_lst)
    ]
    feature_take = ["key_idx", "drugbank_id", "name", "smiles", "QED", "SA", "BA"]
    for i in range(num_pose):
        pose_name = f"pose_{i}_pockets"
        filtered_df[pose_name] = extract_ith_list(pocket_lst, i)
        feature_take.append(pose_name)

    output_docking = filtered_df[feature_take]
    output_docking.iloc[np.argsort(output_docking["BA"])]
    output_docking.to_csv(
        f"/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/docking_output/{protein_name}_{dataset}_pocket_docking_{stage}.csv", index=False
    )

# Function to embed a protein sequence
def embed_protein(sequence, model_name="facebook/esm2_t33_650M_UR50D"):
    model = EsmModel.from_pretrained(model_name)
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        outputs = model(**inputs)
    # Extract the embeddings from the model outputs
    embeddings = outputs.last_hidden_state.squeeze(0).numpy()
    return embeddings

def esm2_embed_protein(protein_name="6dql"):
    # Load the ESM-2 model and tokenizer
    protein_path = f"/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/ligands_conformation/{protein_name}.pdb"
    protein_pdb = Chem.MolFromPDBFile(protein_path, removeHs=False)

    exist = []
    sequence = ""
    for atom in protein_pdb.GetAtoms():
        residue_info = atom.GetPDBResidueInfo()
        if residue_info:
            atom_idx = atom.GetIdx()
            res_name = residue_info.GetResidueName().strip()
            res_num = residue_info.GetResidueNumber()
            chain_id = residue_info.GetChainId()
            pos = protein_pdb.GetConformer().GetAtomPosition(atom_idx)
            if (chain_id, res_name, chain_id) not in exist:
                sequence += three_to_one[res_name]
            exist.append((chain_id, res_name, chain_id))

    embeddings = embed_protein(sequence)
    return embeddings

def get_keepNode(
    com,
    protein_node_xyz,
    n_node,
    pocket_radius,
    use_whole_protein,
    use_compound_com_as_pocket,
    add_noise_to_com,
    chosen_pocket_com,
):
    if use_whole_protein:
        keepNode = np.ones(n_node, dtype=bool)
    else:
        keepNode = np.zeros(n_node, dtype=bool)
        # extract node based on compound COM.
        if use_compound_com_as_pocket:
            if add_noise_to_com:  # com is the mean coordinate of the compound
                com = com + add_noise_to_com * (2 * np.random.rand(*com.shape) - 1)
            dis = euclidean_distances(protein_node_xyz, com.reshape(1, -1))
            keepNode = dis < pocket_radius
    if chosen_pocket_com is not None:
        another_keepNode = np.zeros(n_node, dtype=bool)
        for a_com in chosen_pocket_com:
            if add_noise_to_com:
                a_com = a_com + add_noise_to_com * (
                    2 * np.random.rand(*a_com.shape) - 1
                )
            dis = euclidean_distances(protein_node_xyz, com.reshape(1, -1))
            another_keepNode |= dis < pocket_radius
        keepNode |= another_keepNode
    return keepNode.reshape(-1)


def esm2_new(protein_path):
    parser = PDBParser(QUIET=True)
    s = parser.get_structure("x", protein_path)
    res_list = s.get_residues()

    verbose = False
    ensure_ca_exist = True
    bfactor_cutoff = None
    clean_res_list = ""
    c_alpha_coords = []
    for res in res_list:
        hetero, resid, insertion = res.full_id[-1]
        if hetero == " ":
            if res.resname not in three_to_one:
                if verbose:
                    print(res, "has non-standard resname")
                continue
            if (not ensure_ca_exist) or ("CA" in res):
                if bfactor_cutoff is not None:
                    ca_bfactor = float(res["CA"].bfactor)
                    if ca_bfactor < bfactor_cutoff:
                        continue
                clean_res_list += three_to_one[res.resname]
                c_alpha_coords.append(res["CA"].get_coord())
        else:
            if verbose:
                print(res, res.full_id, "is hetero")
    feature = embed_protein(clean_res_list)
    c_alpha_coords = np.array(c_alpha_coords)
    return torch.tensor(c_alpha_coords), torch.tensor(feature)

def binarize(x):
    return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

# adj - > n_hops connections adj
def n_hops_adj(adj, n_hops):
    adj_mats = [torch.eye(adj.size(0), dtype=torch.long, device=adj.device), binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device))]

    for i in range(2, n_hops+1):
        adj_mats.append(binarize(adj_mats[i-1] @ adj_mats[1]))
    extend_mat = torch.zeros_like(adj)

    for i in range(1, n_hops+1):
        extend_mat += (adj_mats[i] - adj_mats[i-1]) * i

    return extend_mat

def get_LAS_distance_constraint_mask(mol):
    # Get the adj
    adj = Chem.GetAdjacencyMatrix(mol)
    adj = torch.from_numpy(adj)
    extend_adj = n_hops_adj(adj, 2)
    # add ring
    ssr = Chem.GetSymmSSSR(mol)
    for ring in ssr:
        # print(ring)
        for i in ring:
            for j in ring:
                if i == j:
                    continue
                else:
                    extend_adj[i][j] += 1
    # turn to mask
    mol_mask = binarize(extend_adj)
    return mol_mask

def get_compound_pair_dis_distribution(coords, LAS_distance_constraint_mask=None):
    pair_dis = scipy.spatial.distance.cdist(coords, coords)
    bin_size = 1
    bin_min = -0.5
    bin_max = 15
    if LAS_distance_constraint_mask is not None:
        pair_dis[LAS_distance_constraint_mask == 0] = bin_max
        # diagonal is zero.
        for i in range(pair_dis.shape[0]):
            pair_dis[i, i] = 0
    pair_dis = torch.tensor(pair_dis, dtype=torch.float)
    pair_dis[pair_dis > bin_max] = bin_max
    pair_dis_bin_index = torch.div(
        pair_dis - bin_min, bin_size, rounding_mode="floor"
    ).long()
    pair_dis_one_hot = torch.nn.functional.one_hot(pair_dis_bin_index, num_classes=16)
    pair_dis_distribution = pair_dis_one_hot.float()
    return pair_dis_distribution

def extract_torchdrug_feature_from_mol_conformer(ligand_path, index_conformer=0):
    mol = Chem.MolFromPDBFile(ligand_path, removeHs=True)
    coords = mol.GetConformer(index_conformer).GetPositions()
    LAS_distance_constraint_mask = get_LAS_distance_constraint_mask(mol)
    pair_dis_distribution = get_compound_pair_dis_distribution(
        coords, LAS_distance_constraint_mask=LAS_distance_constraint_mask
    )
    molstd = td.Molecule.from_smiles(
        Chem.MolToSmiles(mol), node_feature="property_prediction"
    )
    compound_node_features = molstd.node_feature  # nodes_chemical_features
    edge_list = molstd.edge_list  # [num_edge, 3]
    edge_weight = molstd.edge_weight  # [num_edge, 1]
    assert edge_weight.max() == 1
    assert edge_weight.min() == 1
    assert coords.shape[0] == compound_node_features.shape[0]
    edge_feature = molstd.edge_feature  # [num_edge, edge_feature_dim]
    LAS_distance_edge_index = torch.tensor(np.where(LAS_distance_constraint_mask == 1))
    x = (torch.tensor(coords), torch.tensor(compound_node_features), torch.tensor(edge_list), torch.tensor(edge_feature), torch.tensor(pair_dis_distribution), torch.tensor(LAS_distance_edge_index))
    return x

def extract_torchdrug_database(smiles):
    rdkit_mol = Chem.MolFromSmiles(smiles)
    rdkit_mol = Chem.AddHs(rdkit_mol)
    # Generate a 3D conformation
    AllChem.EmbedMolecule(rdkit_mol)
    # Optimize the 3D conformation
    AllChem.UFFOptimizeMolecule(rdkit_mol)
    mol = Chem.RemoveHs(rdkit_mol)
    coords = mol.GetConformer().GetPositions()
    LAS_distance_constraint_mask = get_LAS_distance_constraint_mask(mol)
    pair_dis_distribution = get_compound_pair_dis_distribution(
        coords, LAS_distance_constraint_mask=LAS_distance_constraint_mask
    )
    molstd = td.Molecule.from_smiles(
        Chem.MolToSmiles(mol), node_feature="property_prediction"
    )
    compound_node_features = molstd.node_feature  # nodes_chemical_features
    edge_list = molstd.edge_list  # [num_edge, 3]
    edge_weight = molstd.edge_weight  # [num_edge, 1]
    assert edge_weight.max() == 1
    assert edge_weight.min() == 1
    assert coords.shape[0] == compound_node_features.shape[0]
    edge_feature = molstd.edge_feature  # [num_edge, edge_feature_dim]
    LAS_distance_edge_index = torch.tensor(np.where(LAS_distance_constraint_mask == 1))
    x = (mol, torch.tensor(coords), torch.tensor(compound_node_features), torch.tensor(edge_list), torch.tensor(edge_feature), 
        torch.tensor(pair_dis_distribution), torch.tensor(LAS_distance_edge_index))
    return x

def uniform_random_rotation(x):
    """Apply a random rotation in 3D, with a distribution uniform over the
    sphere.
    Arguments:
        x: vector or set of vectors with dimension (n, 3), where n is the
            number of vectors
    Returns:
        Array of shape (n, 3) containing the randomly rotated vectors of x,
        about the mean coordinate of x.
    Algorithm taken from "Fast Random Rotation Matrices" (James Avro, 1992):
    https://doi.org/10.1016/B978-0-08-050755-2.50034-8
    """

    def generate_random_z_axis_rotation():
        """Generate random rotation matrix about the z axis."""
        R = np.eye(3)
        x1 = np.random.rand()
        R[0, 0] = R[1, 1] = np.cos(2 * np.pi * x1)
        R[0, 1] = -np.sin(2 * np.pi * x1)
        R[1, 0] = np.sin(2 * np.pi * x1)
        return R

    # There are two random variables in [0, 1) here (naming is same as paper)
    x2 = 2 * np.pi * np.random.rand()
    x3 = np.random.rand()
    # Rotation of all points around x axis using matrix
    R = generate_random_z_axis_rotation()
    v = np.array([np.cos(x2) * np.sqrt(x3), np.sin(x2) * np.sqrt(x3), np.sqrt(1 - x3)])
    H = np.eye(3) - (2 * np.outer(v, v))
    M = -(H @ R)
    x = x.reshape((-1, 3))
    mean_coord = np.mean(x, axis=0)
    return ((x - mean_coord) @ M) + mean_coord @ M

def construct_data_from_graph_gvp_mean(
    args,
    protein_node_xyz,
    coords,
    compound_node_features,
    input_atom_edge_list,
    input_atom_edge_attr_list,
    LAS_edge_index,
    rdkit_coords,
    compound_coords_init_mode="pocket_center_rdkit",
    includeDisMap=True,
    pdb_id=None,
    group="train",
    seed=42,
    data_path=None,
    contactCutoff=8.0,
    pocket_radius=20,
    interactionThresholdDistance=10,
    compoundMode=1,
    add_noise_to_com=None,
    use_whole_protein=False,
    use_compound_com_as_pocket=True,
    chosen_pocket_com=None,
    random_rotation=False,
    pocket_idx_no_noise=True,
    protein_esm2_feat=None,
):
    n_node = protein_node_xyz.shape[0]
    # n_compound_node = coords.shape[0]
    # normalize the protein and ligand coords
    coords_bias = protein_node_xyz.mean(dim=0)
    coords = coords - coords_bias.numpy()
    protein_node_xyz = protein_node_xyz - coords_bias
    # centroid instead of com.
    com = coords.mean(axis=0)
    if args.train_pred_pocket_noise and group == "train":
        keepNode = get_keepNode(
            com,
            protein_node_xyz.numpy(),
            n_node,
            pocket_radius,
            use_whole_protein,
            use_compound_com_as_pocket,
            args.train_pred_pocket_noise,
            chosen_pocket_com,
        )
    else:
        keepNode = get_keepNode(
            com,
            protein_node_xyz.numpy(),
            n_node,
            pocket_radius,
            use_whole_protein,
            use_compound_com_as_pocket,
            add_noise_to_com,
            chosen_pocket_com,
        )

    keepNode_no_noise = get_keepNode(
        com,
        protein_node_xyz.numpy(),
        n_node,
        pocket_radius,
        use_whole_protein,
        use_compound_com_as_pocket,
        None,
        chosen_pocket_com,
    )
    if keepNode.sum() < 5:
        # if only include less than 5 residues, simply add first 100 residues.
        keepNode[:100] = True

    input_node_xyz = protein_node_xyz[keepNode]
    # input_edge_idx, input_protein_edge_s, input_protein_edge_v = get_protein_edge_features_and_index(protein_edge_index, protein_edge_s, protein_edge_v, keepNode)

    # construct heterogeneous graph data.
    data = HeteroData()
    # only if your ligand is real this y_contact is meaningful. Distance map between ligand atoms and protein amino acids.
    dis_map = scipy.spatial.distance.cdist(input_node_xyz.cpu().numpy(), coords)
    # y_contact = dis_map < contactCutoff # contactCutoff is 8A
    if includeDisMap:
        # treat all distance above 10A as the same.
        dis_map[dis_map > interactionThresholdDistance] = interactionThresholdDistance
        data.dis_map = torch.tensor(dis_map, dtype=torch.float).flatten()
    # TODO The difference between contactCutoff and interactionThresholdDistance:
    # contactCutoff is for classification evaluation, interactionThresholdDistance is for distance regression.
    # additional information. keep records.
    data.node_xyz = input_node_xyz
    data.coords = torch.tensor(coords, dtype=torch.float)
    # data.y = torch.tensor(y_contact, dtype=torch.float).flatten() # whether the distance between ligand and protein is less than 8A.

    # pocket information

    if torch.is_tensor(protein_esm2_feat):
        data["pocket"].node_feats = protein_esm2_feat[keepNode]
    else:
        raise ValueError("protein_esm2_feat should be a tensor")

    data["pocket"].keepNode = torch.tensor(keepNode, dtype=torch.bool)

    data["compound"].node_feats = compound_node_features.float()
    data['compound', 'LAS', 'compound'].edge_index = LAS_edge_index
    # print(LAS_edge_index)

    # complex information
    n_protein       = input_node_xyz.shape[0]
    n_protein_whole = protein_node_xyz.shape[0]
    n_compound      = compound_node_features.shape[0]
    # n_protein:            number of interaction protein
    # n_protein_whole:      number of the whole   protein
    # n_compound:           number of atom in the compound
    # use zero coord to init compound
    data['complex'].node_coords = torch.cat( # [glb_c || compound || glb_p || protein]
        (torch.zeros(n_compound + 2, 3), input_node_xyz), dim=0
        ).float()
    if random_rotation:
        rdkit_coords = torch.tensor(uniform_random_rotation(rdkit_coords))
    else:
        rdkit_coords = torch.tensor(rdkit_coords)
    coords_init = (
        rdkit_coords
        - rdkit_coords.mean(dim=0).reshape(1, 3)
        + input_node_xyz.mean(dim=0).reshape(1, 3)
    )

    # ground truth ligand and pocket
    data["complex"].node_coords = torch.cat(  # [glb_c || compound || glb_p || protein]
        (torch.zeros(1, 3), coords_init, torch.zeros(1, 3), input_node_xyz), dim=0
    ).float()

    data["complex"].node_coords_LAS = (
        torch.cat(  # [glb_c || compound || glb_p || protein]
            (
                torch.zeros(1, 3),
                rdkit_coords,
                torch.zeros(1, 3),
                torch.zeros_like(input_node_xyz),
            ),
            dim=0,
        ).float()
    )

    segment = torch.zeros(n_protein + n_compound + 2)
    segment[n_compound + 1 :] = 1  # compound: 0, protein: 1
    data["complex"].segment = segment  # protein or ligand
    mask = torch.zeros(n_protein + n_compound + 2)
    mask[: n_compound + 2] = 1  # glb_p can be updated
    data["complex"].mask = mask.bool()
    is_global = torch.zeros(n_protein + n_compound + 2)
    is_global[0] = 1
    is_global[n_compound + 1] = 1
    data["complex"].is_global = is_global.bool()

    data["complex", "c2c", "complex"].edge_index = (
        input_atom_edge_list[:, :2].long().t().contiguous() + 1
    )
    if (
        compound_coords_init_mode == "redocking"
        or compound_coords_init_mode == "redocking_no_rotate"
    ):
        data["complex", "LAS", "complex"].edge_index = (
            torch.nonzero(torch.ones(n_compound, n_compound)).t() + 1
        )
    else:
        data["complex", "LAS", "complex"].edge_index = LAS_edge_index + 1

    # ground truth ligand and whole protein
    data["complex_whole_protein"].node_coords = (
        torch.cat(  # [glb_c || compound || glb_p || protein]
            (
                torch.zeros(1, 3),
                coords_init
                - coords_init.mean(dim=0).reshape(
                    1, 3
                ),  # for pocket prediction module, the ligand is centered at the protein center/origin
                torch.zeros(1, 3),
                protein_node_xyz,
            ),
            dim=0,
        ).float()
    )

    if (
        compound_coords_init_mode == "redocking"
        or compound_coords_init_mode == "redocking_no_rotate"
    ):
        data["complex_whole_protein"].node_coords_LAS = (
            torch.cat(  # [glb_c || compound || glb_p || protein]
                (
                    torch.zeros(1, 3),
                    torch.tensor(coords),
                    torch.zeros(1, 3),
                    torch.zeros_like(protein_node_xyz),
                ),
                dim=0,
            ).float()
        )
    else:
        data["complex_whole_protein"].node_coords_LAS = (
            torch.cat(  # [glb_c || compound || glb_p || protein]
                (
                    torch.zeros(1, 3),
                    rdkit_coords,
                    torch.zeros(1, 3),
                    torch.zeros_like(protein_node_xyz),
                ),
                dim=0,
            ).float()
        )

    segment = torch.zeros(n_protein_whole + n_compound + 2)
    segment[n_compound + 1 :] = 1  # compound: 0, protein: 1
    data["complex_whole_protein"].segment = segment  # protein or ligand
    mask = torch.zeros(n_protein_whole + n_compound + 2)
    mask[: n_compound + 2] = 1  # glb_p can be updated
    data["complex_whole_protein"].mask = mask.bool()
    is_global = torch.zeros(n_protein_whole + n_compound + 2)
    is_global[0] = 1
    is_global[n_compound + 1] = 1
    data["complex_whole_protein"].is_global = is_global.bool()

    data["complex_whole_protein", "c2c", "complex_whole_protein"].edge_index = (
        input_atom_edge_list[:, :2].long().t().contiguous() + 1
    )
    
    if (
        compound_coords_init_mode == "redocking"
        or compound_coords_init_mode == "redocking_no_rotate"
    ):
        data["complex_whole_protein", "LAS", "complex_whole_protein"].edge_index = (
            torch.nonzero(torch.ones(n_compound, n_compound)).t() + 1
        )
    else:
        data["complex_whole_protein", "LAS", "complex_whole_protein"].edge_index = (
            LAS_edge_index + 1
        )

    # for stage 3
    data["compound"].node_coords = coords_init
    data["compound"].rdkit_coords = rdkit_coords
    data["compound_atom_edge_list"].x = (
        input_atom_edge_list[:, :2].long().contiguous() + 1
    ).clone()
    data["LAS_edge_list"].x = data["complex", "LAS", "complex"].edge_index.clone().t()
    # add whole protein information for pocket prediction

    data.node_xyz_whole = protein_node_xyz
    data.coords_center = torch.tensor(com, dtype=torch.float).unsqueeze(0)
    # data.seq_whole = protein_seq
    data.coord_offset = coords_bias.unsqueeze(0)
    # save the pocket index for binary classification
    if pocket_idx_no_noise:
        data.pocket_idx = torch.tensor(keepNode_no_noise, dtype=torch.int)
    else:
        data.pocket_idx = torch.tensor(keepNode, dtype=torch.int)

    if torch.is_tensor(protein_esm2_feat):
        data["protein_whole"].node_feats = protein_esm2_feat
    else:
        raise ValueError("protein_esm2_feat should be a tensor")
    return data, input_node_xyz, keepNode

def ligand_from_smiles(smiles):
    rdkit_mol = Chem.MolFromSmiles(smiles)
    rdkit_mol = Chem.AddHs(rdkit_mol)
    # Generate a 3D conformation
    AllChem.EmbedMolecule(rdkit_mol)
    # Optimize the 3D conformation
    AllChem.UFFOptimizeMolecule(rdkit_mol)
    rdkit_mol = Chem.RemoveHs(rdkit_mol)
    # Retrieve the conformer
    conformer = rdkit_mol.GetConformer()
    return conformer.GetPositions()

def mol_from_smiles(smiles):
    rdkit_mol = Chem.MolFromSmiles(smiles)
    rdkit_mol = Chem.AddHs(rdkit_mol)
    # Generate a 3D conformation
    AllChem.EmbedMolecule(rdkit_mol)
    # Optimize the 3D conformation
    AllChem.UFFOptimizeMolecule(rdkit_mol)
    rdkit_mol = Chem.RemoveHs(rdkit_mol)
    return rdkit_mol

def create_inference_data(protein_name = '1a01', database_name = "drugbank"):
    if os.path.exists(f"/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/inference_data/inference_{protein_name}_{database_name}.pt"):
        return 
    protein_node_xyz, protein_esm_feature = esm2_new(protein_name, '/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/pdb')
    compound_paths = sorted(glob(f'/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/ligands_conformation/{database_name}/*'), key = lambda x: int(os.path.split(x)[1].split('.')[0]))
    database_df    = pd.read_csv(f"/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/data/{database_name}.csv")
    index_data     = 0
    inference_data = []
    for compound_path in tqdm(compound_paths):
        try:
            idx = int(os.path.split(compound_path)[1].split('.')[0])
            smiles = database_df['smiles'][idx]
            (
                mol,
                rdkit_coords,
                compound_node_features,
                input_atom_edge_list,
                input_atom_edge_attr_list,
                pair_dis_distribution,
                LAS_edge_index,
            ) = extract_torchdrug_database(smiles)
            n_protein_whole = protein_node_xyz.shape[0]
            n_compound = compound_node_features.shape[0]
            data = HeteroData()
            data.coord_offset = protein_node_xyz.mean(dim=0).unsqueeze(0)
            protein_node_xyz  = protein_node_xyz - protein_node_xyz.mean(dim=0)
            coords_init       = rdkit_coords - rdkit_coords.mean(axis=0)
            data['compound'].node_feats = compound_node_features.float()
            data['compound', 'LAS', 'compound'].edge_index = LAS_edge_index
            data['compound'].node_coords = coords_init
            data['compound'].rdkit_coords = coords_init
            data['compound'].smiles = smiles
            data['compound_atom_edge_list'].x = (input_atom_edge_list[:,:2].long().contiguous() + 1).clone()
            data['LAS_edge_list'].x = (LAS_edge_index + 1).clone().t()
            data.node_xyz_whole = protein_node_xyz
            data.idx = index_data
            data.uid = protein_name
            data.mol = mol
            data.ligand_id = idx
            data['complex_whole_protein'].node_coords = torch.cat( # [glb_c || compound || glb_p || protein]
            (
                torch.zeros(1, 3),
                coords_init - coords_init.mean(dim=0), # for pocket prediction module, the ligand is centered at the protein center/origin
                torch.zeros(1, 3), 
                protein_node_xyz
            ), dim=0).float()
            data['complex_whole_protein'].node_coords_LAS = torch.cat( # [glb_c || compound || glb_p || protein]
                (
                    torch.zeros(1, 3),
                    rdkit_coords,
                    torch.zeros(1, 3), 
                    torch.zeros_like(protein_node_xyz)
                ), dim=0
            ).float()

            segment = torch.zeros(n_protein_whole + n_compound + 2)
            segment[n_compound+1:] = 1 # compound: 0, protein: 1
            data['complex_whole_protein'].segment = segment # protein or ligand
            mask = torch.zeros(n_protein_whole + n_compound + 2)
            mask[:n_compound+2] = 1 # glb_p can be updated
            data['complex_whole_protein'].mask = mask.bool()
            is_global = torch.zeros(n_protein_whole + n_compound + 2)
            is_global[0] = 1
            is_global[n_compound+1] = 1
            data['complex_whole_protein'].is_global = is_global.bool()

            data['complex_whole_protein', 'c2c', 'complex_whole_protein'].edge_index = input_atom_edge_list[:,:2].long().t().contiguous() + 1
            data['complex_whole_protein', 'LAS', 'complex_whole_protein'].edge_index = LAS_edge_index + 1

            data['protein_whole'].node_feats = protein_esm_feature
            index_data += 1
            inference_data.append(data)
        except:
            continue
    print(len(inference_data))
    torch.save(inference_data, f"/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/inference_data/inference_{protein_name}_{database_name}.pt")
    del inference_data

def get_inference_data(protein_name = "1a01", database="drugbank"):
    path = f'/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/inference_data/inference_{protein_name}_{database}.pt'
    if not os.path.exists(path):
        create_inference_data(protein_name, database)
    return torch.load(path)


def get_simulate_data(args, protein_path , dataset_path , group = "train"):
    protein_node_xyz, protein_esm2_feat = esm2_new(protein_path)

    print(protein_path)
    exit()
    compound_conformer_id = 0
    df = pd.read_csv(dataset_path)
    compound_lst = glob(
       '/cm/archive/phucpht/Drug_research/Blind_pocket/datasets/dude_process/aa2ar/binding_data/*.pdbqt'
    ) + glob(
       '/cm/archive/phucpht/Drug_research/Blind_pocket/datasets/dude_process/aa2ar/generation_docking/*.pdbqt'
    ) 


    lst_data = []
    for idx, compound_path in enumerate(compound_lst):
        try:
            compound_name = os.path.split(compound_path)[1].split(".")[0]
            (
                coords,
                compound_node_features,
                input_atom_edge_list,
                input_atom_edge_attr_list,
                pair_dis_distribution,
                LAS_edge_index,
            ) = extract_torchdrug_feature_from_mol_conformer(
                compound_path, compound_conformer_id
            )
            exit()
            
            smiles = df["smiles"][int(compound_name)]
            rdkit_coords = ligand_from_smiles(smiles)
            
            hereto_data, _, _ = construct_data_from_graph_gvp_mean(
                args,
                torch.tensor(protein_node_xyz),
                coords,
                torch.tensor(compound_node_features),
                input_atom_edge_list,
                input_atom_edge_attr_list,
                LAS_edge_index,
                rdkit_coords,
                compound_coords_init_mode="pocket_center_rdkit",
                includeDisMap=True,
                pdb_id=None,
                group=group,
                seed=42,
                data_path=None,
                pocket_radius=20,
                interactionThresholdDistance=10,
                add_noise_to_com=None,
                use_whole_protein=False,
                use_compound_com_as_pocket=True,
                chosen_pocket_com=None,
                random_rotation=False,
                pocket_idx_no_noise=True,
                protein_esm2_feat=protein_esm2_feat,
            )
            lst_data.append(hereto_data)
        except:
            continue
    return lst_data

def  arg_parsing_fix(): 
    parser = argparse.ArgumentParser(description='FABind model training.')

    parser.add_argument("-m", "--mode", type=int, default=0,
                help="mode specify the model to use.")
    parser.add_argument("-d", "--data", type=str, default="0",
                help="data specify the data to use. \
                0 for re-docking, 1 for self-docking.")
    parser.add_argument('--seed', type=int, default=42,
                help="seed to use.")
    parser.add_argument("--gs-tau", type=float, default=1,
                help="Tau for the temperature-based softmax.")
    parser.add_argument("--gs-hard", action='store_true', default=False,
                help="Hard mode for gumbel softmax.")
    parser.add_argument("--batch_size", type=int, default=8,
                help="batch size.")
    parser.add_argument("--restart", type=str, default=None,
                help="continue the training from the model we saved from scratch.")
    parser.add_argument("--reload", type=str, default=None,
                help="continue the training from the model we saved.")
    parser.add_argument("--addNoise", type=str, default=None,
                help="shift the location of the pocket center in each training sample \
                such that the protein pocket encloses a slightly different space.")

    pair_interaction_mask = parser.add_mutually_exclusive_group()
    # use_equivalent_native_y_mask is probably a better choice.
    pair_interaction_mask.add_argument("--use_y_mask", action='store_true', default=False,
                help="mask the pair interaction during pair interaction loss evaluation based on data.real_y_mask. \
                real_y_mask=True if it's the native pocket that ligand binds to.")
    pair_interaction_mask.add_argument("--use_equivalent_native_y_mask", action='store_true', default=False,
                help="mask the pair interaction during pair interaction loss evaluation based on data.equivalent_native_y_mask. \
                real_y_mask=True if most of the native interaction between ligand and protein happen inside this pocket.")
    parser.add_argument("--use_affinity_mask", type=int, default=0,
                help="mask affinity in loss evaluation based on data.real_affinity_mask")
    parser.add_argument("--affinity_loss_mode", type=int, default=1,
                help="define which affinity loss function to use.")
    parser.add_argument("--pred_dis", type=int, default=1,
                help="pred distance map or predict contact map.")
    parser.add_argument("--posweight", type=int, default=8,
                help="pos weight in pair contact loss, not useful if args.pred_dis=1")
    parser.add_argument("--relative_k", type=float, default=0.01,
                help="adjust the strength of the affinity loss head relative to the pair interaction loss.")
    parser.add_argument("-r", "--relative_k_mode", type=int, default=0,
                help="define how the relative_k changes over epochs")
    parser.add_argument("--resultFolder", type=str, default="/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/result",
                help="information you want to keep a record.")
    parser.add_argument("--label", type=str, default="",
                help="information you want to keep a record.")
    parser.add_argument("--use-whole-protein", action='store_true', default=False,
                help="currently not used.")
    parser.add_argument("--data-path", type=str, default="/PDBbind_data/pdbbind2020",
                help="Data path.")
    parser.add_argument("--exp-name", type=str, default="",
                help="data path.")
    parser.add_argument("--tqdm-interval", type=float, default=0.1,
                help="tqdm bar update interval")
    parser.add_argument("--lr", type=float, default=0.0001)

    parser.add_argument("--pocket-coord-huber-delta", type=float, default=3.0)

    parser.add_argument("--coord-loss-function", type=str, default='SmoothL1', choices=['MSE', 'SmoothL1'])

    parser.add_argument("--coord-loss-weight", type=float, default=1.0)
    parser.add_argument("--pair-distance-loss-weight", type=float, default=1.0)
    parser.add_argument("--pair-distance-distill-loss-weight", type=float, default=1.0)
    parser.add_argument("--pocket-cls-loss-weight", type=float, default=1.0)
    parser.add_argument("--pocket-distance-loss-weight", type=float, default=0.05)
    parser.add_argument("--pocket-cls-loss-func", type=str, default='bce')

    # parser.add_argument("--warm-mae-thr", type=float, default=5.0)

    parser.add_argument("--use-compound-com-cls", action='store_true', default=False,
                help="only use real pocket to run pocket classification task")

    parser.add_argument("--compound-coords-init-mode", type=str, default="pocket_center_rdkit",
                choices=['pocket_center_rdkit', 'pocket_center', 'compound_center', 'perturb_3A', 'perturb_4A', 'perturb_5A', 'random'])

    parser.add_argument('--trig-layers', type=int, default=1)

    parser.add_argument('--distmap-pred', type=str, default='mlp',
                choices=['mlp', 'trig'])
    parser.add_argument('--mean-layers', type=int, default=3)
    parser.add_argument('--n-iter', type=int, default=5)
    parser.add_argument('--inter-cutoff', type=float, default=10.0)
    parser.add_argument('--intra-cutoff', type=float, default=8.0)
    parser.add_argument('--refine', type=str, default='refine_coord',
                choices=['stack', 'refine_coord'])

    parser.add_argument('--coordinate-scale', type=float, default=5.0)
    parser.add_argument('--geometry-reg-step-size', type=float, default=0.001)
    parser.add_argument('--lr-scheduler', type=str, default="constant", choices=['constant', 'poly_decay', 'cosine_decay', 'cosine_decay_restart', 'exp_decay'])

    parser.add_argument('--add-attn-pair-bias', action='store_true', default=False)
    parser.add_argument('--explicit-pair-embed', action='store_true', default=False)
    parser.add_argument('--opm', action='store_true', default=False)

    parser.add_argument('--add-cross-attn-layer', action='store_true', default=False)
    parser.add_argument('--rm-layernorm', action='store_true', default=False)
    parser.add_argument('--keep-trig-attn', action='store_true', default=False)

    parser.add_argument('--pocket-radius', type=float, default=20.0)

    parser.add_argument('--rm-LAS-constrained-optim', action='store_true', default=False)
    parser.add_argument('--rm-F-norm', action='store_true', default=False)
    parser.add_argument('--norm-type', type=str, default="per_sample", choices=['per_sample', '4_sample', 'all_sample'])

    # parser.add_argument("--only-predicted-pocket-mae-thr", type=float, default=3.0)
    parser.add_argument('--noise-for-predicted-pocket', type=float, default=5.0)
    parser.add_argument('--test-random-rotation', action='store_true', default=False)

    parser.add_argument('--random-n-iter', action='store_true', default=False)
    parser.add_argument('--clip-grad', action='store_true', default=False)

    # one batch actually contains 20000 samples, not the size of training set
    parser.add_argument("--sample-n", type=int, default=0, help="number of samples in one epoch.")

    parser.add_argument('--fix-pocket', action='store_true', default=False)
    parser.add_argument('--pocket-idx-no-noise', action='store_true', default=False)
    parser.add_argument('--ablation-no-attention', action='store_true', default=False)
    parser.add_argument('--ablation-no-attention-with-cross-attn', action='store_true', default=False)

    parser.add_argument('--redocking', action='store_true', default=False)
    parser.add_argument('--redocking-no-rotate', action='store_true', default=False)

    parser.add_argument("--pocket-pred-layers", type=int, default=1, help="number of layers for pocket pred model.")
    parser.add_argument('--pocket-pred-n-iter', type=int, default=1, help="number of iterations for pocket pred model.")

    parser.add_argument('--use-esm2-feat', action='store_true', default=False)
    parser.add_argument("--center-dist-threshold", type=float, default=8.0)

    parser.add_argument("--mixed-precision", type=str, default='no', choices=['no', 'fp16'])
    parser.add_argument('--disable-tqdm', action='store_true', default=False)
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--optim', type=str, default='adam', choices=['adam', 'adamw'])
    parser.add_argument("--warmup-epochs", type=int, default=15,
                help="used in combination with relative_k_mode.")
    parser.add_argument("--total-epochs", type=int, default=400,
                help="option to switch training data after certain epochs.")
    parser.add_argument('--disable-validate', action='store_true', default=False)
    parser.add_argument('--disable-tensorboard', action='store_true', default=False)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--stage-prob", type=float, default=0.5)
    parser.add_argument("--pocket-pred-hidden-size", type=int, default=128)

    parser.add_argument("--local-eval", action='store_true', default=False)
    parser.add_argument("--train-ligand-torsion-noise", action='store_true', default=False)
    parser.add_argument("--train-pred-pocket-noise", type=float, default=0.0)
    parser.add_argument('--esm2-concat-raw', action='store_true', default=False)
    parser.add_argument('--ckpt-path', default='/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/fabind/ckpt/best_model.bin')
    args = parser.parse_args("")
    args.mode=5
    args.data='0'
    args.seed=42
    args.gs_tau=1
    args.gs_hard=False
    args.batch_size=8
    args.restart=None
    args.reload=None
    args.addNoise='5'
    args.use_y_mask=False
    args.use_equivalent_native_y_mask=False
    args.use_affinity_mask=0
    args.affinity_loss_mode=1
    args.pred_dis=1
    args.posweight=8
    args.relative_k=0.01
    args.relative_k_mode=0
    args.resultFolder='/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/results'
    args.label='baseline'
    args.use_whole_protein=False
    args.data_path='pdbbind2020'
    args.exp_name='train_tmp'
    args.tqdm_interval=0.1
    args.lr=5e-05
    args.pocket_coord_huber_delta=3.0
    args.coord_loss_function='SmoothL1'
    args.coord_loss_weight=1.0
    args.pair_distance_loss_weight=1.0
    args.pair_distance_distill_loss_weight=1.0
    args.pocket_cls_loss_weight=1.0
    args.pocket_distance_loss_weight=0.05
    args.pocket_cls_loss_func='bce'
    args.use_compound_com_cls=True
    args.compound_coords_init_mode='pocket_center_rdkit'
    args.trig_layers=1
    args.distmap_pred='mlp'
    args.mean_layers=4
    args.n_iter=8
    args.inter_cutoff=10.0
    args.intra_cutoff=8.0
    args.refine='refine_coord'
    args.coordinate_scale=5.0
    args.geometry_reg_step_size=0.001
    args.lr_scheduler='poly_decay'
    args.add_attn_pair_bias=True
    args.explicit_pair_embed=True
    args.opm=False
    args.add_cross_attn_layer=True
    args.rm_layernorm=True
    args.keep_trig_attn=False
    args.pocket_radius=20.0
    args.rm_LAS_constrained_optim=False
    args.rm_F_norm=False
    args.norm_type='per_sample'
    args.noise_for_predicted_pocket=0.0
    args.test_random_rotation=False
    args.random_n_iter=True
    args.clip_grad=True
    args.sample_n=0
    args.fix_pocket=False
    args.pocket_idx_no_noise=True
    args.ablation_no_attention=False
    args.ablation_no_attention_with_cross_attn=False
    args.redocking=False
    args.redocking_no_rotate=False
    args.pocket_pred_layers=1
    args.pocket_pred_n_iter=1
    args.use_esm2_feat=True
    args.center_dist_threshold=8.0
    args.mixed_precision='no'
    args.disable_tqdm=False
    args.log_interval=100
    args.optim='adam'
    args.warmup_epochs=15
    args.total_epochs=20
    args.disable_validate=False
    args.disable_tensorboard=False
    args.hidden_size=512
    args.weight_decay=0.0
    args.stage_prob=0.5
    args.pocket_pred_hidden_size=128
    args.local_eval=False
    args.train_ligand_torsion_noise=False
    args.train_pred_pocket_noise=0.0
    args.esm2_concat_raw=False

    return args

def arg_parsing_inference():
    parser = argparse.ArgumentParser(description='Train your own TankBind model.')

    parser.add_argument("-m", "--mode", type=int, default=0,
                        help="mode specify the model to use.")
    parser.add_argument("-d", "--data", type=str, default="0",
                        help="data specify the data to use. \
                        0 for re-docking, 1 for self-docking.")
    parser.add_argument('--seed', type=int, default=42,
                        help="seed to use.")
    parser.add_argument("--gs-tau", type=float, default=1,
                        help="Tau for the temperature-based softmax.")
    parser.add_argument("--gs-hard", action='store_true', default=False,
                        help="Hard mode for gumbel softmax.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="batch size.")

    parser.add_argument("--restart", type=str, default=None,
                        help="continue the training from the model we saved from scratch.")
    parser.add_argument("--reload", type=str, default=None,
                        help="continue the training from the model we saved.")
    parser.add_argument("--addNoise", type=str, default=None,
                        help="shift the location of the pocket center in each training sample \
                        such that the protein pocket encloses a slightly different space.")

    pair_interaction_mask = parser.add_mutually_exclusive_group()
    # use_equivalent_native_y_mask is probably a better choice.
    pair_interaction_mask.add_argument("--use_y_mask", action='store_true', default=False,
                        help="mask the pair interaction during pair interaction loss evaluation based on data.real_y_mask. \
                        real_y_mask=True if it's the native pocket that ligand binds to.")
    pair_interaction_mask.add_argument("--use_equivalent_native_y_mask", action='store_true', default=False,
                        help="mask the pair interaction during pair interaction loss evaluation based on data.equivalent_native_y_mask. \
                        real_y_mask=True if most of the native interaction between ligand and protein happen inside this pocket.")

    parser.add_argument("--use_affinity_mask", type=int, default=0,
                        help="mask affinity in loss evaluation based on data.real_affinity_mask")
    parser.add_argument("--affinity_loss_mode", type=int, default=1,
                        help="define which affinity loss function to use.")

    parser.add_argument("--pred_dis", type=int, default=1,
                        help="pred distance map or predict contact map.")
    parser.add_argument("--posweight", type=int, default=8,
                        help="pos weight in pair contact loss, not useful if args.pred_dis=1")

    parser.add_argument("--relative_k", type=float, default=0.01,
                        help="adjust the strength of the affinity loss head relative to the pair interaction loss.")
    parser.add_argument("-r", "--relative_k_mode", type=int, default=0,
                        help="define how the relative_k changes over epochs")

    parser.add_argument("--resultFolder", type=str, default="/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/result",
                        help="information you want to keep a record.")
    parser.add_argument("--label", type=str, default="",
                        help="information you want to keep a record.")

    parser.add_argument("--use-whole-protein", action='store_true', default=False,
                        help="currently not used.")

    parser.add_argument("--data-path", type=str, default="",
                        help="Data path.")
                        
    parser.add_argument("--exp-name", type=str, default="",
                        help="data path.")

    parser.add_argument("--tqdm-interval", type=float, default=0.1,
                        help="tqdm bar update interval")

    parser.add_argument("--lr", type=float, default=0.0001)

    parser.add_argument("--pocket-coord-huber-delta", type=float, default=3.0)

    parser.add_argument("--coord-loss-function", type=str, default='SmoothL1', choices=['MSE', 'SmoothL1'])

    parser.add_argument("--coord-loss-weight", type=float, default=1.0)
    parser.add_argument("--pair-distance-loss-weight", type=float, default=1.0)
    parser.add_argument("--pair-distance-distill-loss-weight", type=float, default=1.0)
    parser.add_argument("--pocket-cls-loss-weight", type=float, default=1.0)
    parser.add_argument("--pocket-distance-loss-weight", type=float, default=0.05)
    parser.add_argument("--pocket-cls-loss-func", type=str, default='bce', choices=['bce', 'dice'])

    # parser.add_argument("--warm-mae-thr", type=float, default=5.0)

    parser.add_argument("--use-compound-com-cls", action='store_true', default=False,
                        help="only use real pocket to run pocket classification task")

    parser.add_argument("--compound-coords-init-mode", type=str, default="pocket_center_rdkit",
                        choices=['pocket_center_rdkit', 'pocket_center', 'compound_center', 'perturb_3A', 'perturb_4A', 'perturb_5A', 'random', 'diffdock'])

    parser.add_argument('--trig-layers', type=int, default=1)

    parser.add_argument('--distmap-pred', type=str, default='mlp',
                        choices=['mlp', 'trig'])
    parser.add_argument('--mean-layers', type=int, default=3)
    parser.add_argument('--n-iter', type=int, default=8)
    parser.add_argument('--inter-cutoff', type=float, default=10.0)
    parser.add_argument('--intra-cutoff', type=float, default=8.0)
    parser.add_argument('--refine', type=str, default='refine_coord',
                        choices=['stack', 'refine_coord'])

    parser.add_argument('--coordinate-scale', type=float, default=5.0)
    parser.add_argument('--geometry-reg-step-size', type=float, default=0.001)
    parser.add_argument('--lr-scheduler', type=str, default="constant", choices=['constant', 'poly_decay', 'cosine_decay', 'cosine_decay_restart', 'exp_decay'])

    parser.add_argument('--add-attn-pair-bias', action='store_true', default=False)
    parser.add_argument('--explicit-pair-embed', action='store_true', default=False)
    parser.add_argument('--opm', action='store_true', default=False)

    parser.add_argument('--add-cross-attn-layer', action='store_true', default=False)
    parser.add_argument('--rm-layernorm', action='store_true', default=False)
    parser.add_argument('--keep-trig-attn', action='store_true', default=False)

    parser.add_argument('--pocket-radius', type=float, default=20.0)

    parser.add_argument('--rm-LAS-constrained-optim', action='store_true', default=False)
    parser.add_argument('--rm-F-norm', action='store_true', default=False)
    parser.add_argument('--norm-type', type=str, default="per_sample", choices=['per_sample', '4_sample', 'all_sample'])

    # parser.add_argument("--only-predicted-pocket-mae-thr", type=float, default=3.0)
    parser.add_argument('--noise-for-predicted-pocket', type=float, default=5.0)
    parser.add_argument('--test-random-rotation', action='store_true', default=False)

    parser.add_argument('--random-n-iter', action='store_true', default=False)
    parser.add_argument('--clip-grad', action='store_true', default=False)

    # one batch actually contains 20000 samples, not the size of training set
    parser.add_argument("--sample-n", type=int, default=0, help="number of samples in one epoch.")

    parser.add_argument('--fix-pocket', action='store_true', default=False)
    parser.add_argument('--pocket-idx-no-noise', action='store_true', default=False)
    parser.add_argument('--ablation-no-attention', action='store_true', default=False)
    parser.add_argument('--ablation-no-attention-with-cross-attn', action='store_true', default=False)

    parser.add_argument('--redocking', action='store_true', default=False)
    parser.add_argument('--redocking-no-rotate', action='store_true', default=False)

    parser.add_argument("--pocket-pred-layers", type=int, default=1, help="number of layers for pocket pred model.")
    parser.add_argument('--pocket-pred-n-iter', type=int, default=1, help="number of iterations for pocket pred model.")

    parser.add_argument('--use-esm2-feat', action='store_true', default=False)
    parser.add_argument("--center-dist-threshold", type=float, default=8.0)

    parser.add_argument("--mixed-precision", type=str, default='no', choices=['no', 'fp16'])
    parser.add_argument('--disable-tqdm', action='store_true', default=False)
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--optim', type=str, default='adam', choices=['adam', 'adamw'])
    parser.add_argument("--warmup-epochs", type=int, default=15,
                        help="used in combination with relative_k_mode.")
    parser.add_argument("--total-epochs", type=int, default=400,
                        help="option to switch training data after certain epochs.")
    parser.add_argument('--disable-validate', action='store_true', default=False)
    parser.add_argument('--disable-tensorboard', action='store_true', default=False)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--stage-prob", type=float, default=0.5)
    parser.add_argument("--pocket-pred-hidden-size", type=int, default=128)

    parser.add_argument("--local-eval", action='store_true', default=False)
    # parser.add_argument("--eval-dir", type=str, default=None)

    parser.add_argument("--train-ligand-torsion-noise", action='store_true', default=False)
    parser.add_argument("--train-pred-pocket-noise", type=float, default=0.0)
    parser.add_argument("--esm2-concat-raw", action='store_true', default=False)
    parser.add_argument("--test-sample-n", type=int, default=1)
    parser.add_argument("--return-hidden", action='store_true', default=False)
    parser.add_argument("--confidence-task", type=str, default='classification', choices=['classification', 'regression', 'perfect'])
    parser.add_argument("--confidence-rmsd-thr", type=float, default=2.0)
    parser.add_argument("--confidence-thr", type=float, default=0.5)

    parser.add_argument("--post-optim", action='store_true', default=False)
    parser.add_argument('--post-optim-mode', type=int, default=0)
    parser.add_argument('--post-optim-epoch', type=int, default=1000)
    parser.add_argument("--rigid", action='store_true', default=False)

    parser.add_argument("--ensemble", action='store_true', default=False)
    parser.add_argument("--confidence", action='store_true', default=False)
    parser.add_argument("--test-gumbel-soft", action='store_true', default=False)
    parser.add_argument("--test-pocket-noise", type=float, default=5)
    parser.add_argument("--test-unseen", action='store_true', default=False)

    parser.add_argument('--sdf-output-path-post-optim', type=str, default="")
    parser.add_argument('--write-mol-to-file', action='store_true', default=False)
    parser.add_argument('--sdf-to-mol2', action='store_true', default=False)

    parser.add_argument('--index-csv', type=str, default=None)
    parser.add_argument('--pdb-file-dir', type=str, default="")
    parser.add_argument('--preprocess-dir', type=str, default="")
    parser.add_argument("--ckpt", type=str, default='/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/ckpt/finetune_fabind_6_1.7848972546748625.bin/scheduler.bin')

    arg = parser.parse_args("")
    arg.mode=5
    arg.data='0'
    arg.seed=128
    arg.gs_tau=1
    arg.gs_hard=False
    arg.batch_size=8
    arg.restart=None
    arg.reload=None
    arg.addNoise='5'
    arg.use_y_mask=False
    arg.use_equivalent_native_y_mask=False
    arg.use_affinity_mask=0
    arg.affinity_loss_mode=1
    arg.pred_dis=1
    arg.posweight=8
    arg.relative_k=0.01
    arg.relative_k_mode=0
    arg.resultFolder='/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/result'
    arg.label='baseline'
    arg.use_whole_protein=False
    arg.data_path=''
    arg.exp_name=''
    arg.tqdm_interval=0.1
    arg.lr=0.0001
    arg.pocket_coord_huber_delta=3.0
    arg.coord_loss_function='SmoothL1'
    arg.coord_loss_weight=1.0
    arg.pair_distance_loss_weight=1.0
    arg.pair_distance_distill_loss_weight=1.0
    arg.pocket_cls_loss_weight=1.0
    arg.pocket_distance_loss_weight=0.05
    arg.pocket_cls_loss_func='bce'
    arg.use_compound_com_cls=True
    arg.compound_coords_init_mode='redocking'
    arg.trig_layers=1
    arg.distmap_pred='mlp'
    arg.mean_layers=4
    arg.n_iter=8
    arg.inter_cutoff=10.0
    arg.intra_cutoff=8.0
    arg.refine='refine_coord'
    arg.coordinate_scale=5.0
    arg.geometry_reg_step_size=0.001
    arg.lr_scheduler='constant'
    arg.add_attn_pair_bias=True
    arg.explicit_pair_embed=True
    arg.opm=False
    arg.add_cross_attn_layer=True
    arg.rm_layernorm=True
    arg.keep_trig_attn=False
    arg.pocket_radius=20.0
    arg.rm_LAS_constrained_optim=False
    arg.rm_F_norm=False
    arg.norm_type='per_sample'
    arg.noise_for_predicted_pocket=0.0
    arg.test_random_rotation=False
    arg.random_n_iter=True
    arg.clip_grad=True
    arg.sample_n=0
    arg.fix_pocket=False
    arg.pocket_idx_no_noise=True
    arg.ablation_no_attention=False
    arg.ablation_no_attention_with_cross_attn=False
    arg.redocking=True
    arg.redocking_no_rotate=False
    arg.pocket_pred_layers=1
    arg.pocket_pred_n_iter=1
    arg.use_esm2_feat=True
    arg.center_dist_threshold=4.0
    arg.mixed_precision='no'
    arg.disable_tqdm=False
    arg.log_interval=50
    arg.optim='adamw'
    arg.warmup_epochs=15
    arg.total_epochs=400
    arg.disable_validate=True
    arg.disable_tensorboard=False
    arg.hidden_size=512
    arg.weight_decay=0.01
    arg.stage_prob=0.25
    arg.pocket_pred_hidden_size=128
    arg.local_eval=False
    arg.train_ligand_torsion_noise=False
    arg.train_pred_pocket_noise=0.0
    arg.esm2_concat_raw=False
    arg.test_sample_n=1
    arg.return_hidden=False
    arg.confidence_task='classification'
    arg.confidence_rmsd_thr=2.0
    arg.confidence_thr=0.5
    arg.post_optim=True
    arg.post_optim_mode=0
    arg.post_optim_epoch=1000
    arg.rigid=False
    arg.ensemble=False
    arg.confidence=False
    arg.test_gumbel_soft=True
    arg.test_pocket_noise=5
    arg.test_unseen=False
    arg.sdf_output_path_post_optim='./cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/inference_examples/inference_output'
    arg.write_mol_to_file=True
    arg.sdf_to_mol2=True
    arg.index_csv=None
    arg.pdb_file_dir=''
    arg.preprocess_dir=''
    arg.ckpt='/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/ckpt/finetune_fabind_12_1.7577800689599452_weight.pth'
    return arg

def create_inference_database(database="drugbank"):
    database_frame = pd.read_csv(f"/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/data/{database}.csv") 
    for idx in range(len(database_frame['smiles'])):
        try: 
            molecule = mol_from_smiles(database_frame['smiles'][idx])
            out_folder = f"/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/ligands_conformation/{database}"
            pdb_filename = f'{idx}.pdb'
            out_file = create_and_merge_path(out_folder, pdb_filename)
            with open(out_file, 'w') as pdb_file:
                pdb_file.write(Chem.MolToPDBBlock(molecule))
        except:
            print(idx)

def multi_objective_ranking_2(dataset= "drugbank"):
    protein_paths    = glob('/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/ligands_conformation/*.pdb')
    data_frame       = pd.read_csv(f"/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/data/{dataset}.csv")
    for protein_path in protein_paths:
        protein_name    = os.path.split(protein_path)[1].split('.')[0]
        distance        = torch.load(f"/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/inference_data/{protein_name}_{dataset}_distance_lst_finetune.pt").cpu().numpy()
        index_lst       = torch.load(f"/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/inference_data/{protein_name}_{dataset}_index_lst_finetune.pt").cpu().numpy()
        data_frame[f"Distance_to_{protein_name}"] = np.ones(len(data_frame))*20.0
        data_frame[f"Distance_to_{protein_name}"][index_lst] = distance
    data_frame[f"Sum_dist"] = data_frame['Distance_to_5dl2'] + data_frame['Distance_to_6dql']
    data_frame.dropna()
    df_sorted = data_frame.sort_values(by=['Sum_dist', 'Distance_to_6dql','Distance_to_5dl2'])
    df_sorted_dist = df_sorted[['Sum_dist', 'Distance_to_6dql','Distance_to_5dl2']]
    mask_all = paretoset(df_sorted_dist, sense=["min", "min", "min"])
    paretoset_all = df_sorted[mask_all]
    if not os.path.exists("/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/multi_protein_fabind_result/"):
        os.makedirs("/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/multi_protein_fabind_result/")
    paretoset_all.index.name = 'key_idx'
    paretoset_all.to_csv(f"/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/multi_protein_fabind_result/{dataset}.csv", index = True)
    paretoset_all = pd.read_csv(f"/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/multi_protein_fabind_result/{dataset}.csv")
    return paretoset_all

def docking_2(protein_name="6dql", dataset="drugbank"):
    filtered_ligands = multi_objective_ranking_2(dataset= "drugbank")
    receptor_file_path = f"/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/protein_docking/{protein_name}.pdbqt"
    protein_df = fo.bp_to_df(fo.read_pdb(receptor_file_path))
    center_x, center_y, center_z = (
        np.mean(protein_df["x"]),
        np.mean(protein_df["y"]),
        np.mean(protein_df["z"]),
    )
    func = Vina_3d(
        receptor_file_path,
        [float(center_x), float(center_y), float(center_z)],
        [35.0, 35.0, 35.0],
    )
    ba_lst = np.ones(len(filtered_ligands["smiles"]))
    
    for idx, smiles in enumerate(filtered_ligands["smiles"]):
        try:
            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), True)
            LigPrepper.smiles2pdbqt(smiles, labels=f"generate_{protein_name}")
            out_folder = f"/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/docking_output/{protein_name}_{dataset}_binding_ligands_stage_2"
            name_out_ligands = str(filtered_ligands["key_idx"][idx])
            out_ligand = create_and_merge_path(
                out_folder, f"{str(name_out_ligands)}.pdbqt"
            )
            ba_generation = func(f"generate_{protein_name}.pdbqt", out_ligand)
            ba_lst[idx] = ba_generation
        except:
            ba_lst[idx] = 0.0
        filtered_ligands["BA"] = ba_lst
        filtered_ligands.to_csv(
            f"/cm/shared/duynvt3/Molecule_Generation/Drug_repurposing/DrugBank/docking_output/{protein_name}_{dataset}_docking2.csv", index=False
        )
        os.remove(f"generate_{protein_name}.pdbqt")

def sdf_to_smiles(sdf_file):
    suppl = Chem.SDMolSupplier(sdf_file)
    smiles_lst = []
    for mol in suppl:
        if mol is not None:
            smiles = Chem.MolToSmiles(mol)
            smiles_lst.append(smiles)
    return smiles_lst