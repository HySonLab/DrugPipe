import os
import sys
from utils_dude import *
from fabind.utils.metrics import *
from fabind.utils.utils import *
import sys
from fabind.models.model import *
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import set_seed
import subprocess
import shutil
from vina import Vina
import warnings
import pdbreader
warnings.filterwarnings("ignore")

v = Vina(sf_name='vina')
user_name = 'phucpht'
device = torch.device("cuda:0")
# ====== Global variables ======
folder_paths =  sorted(glob(f"datasets/dude_process/*"))
idx = int(sys.argv[1])
folder_path = folder_paths[idx]
protein_name = os.path.split(folder_path)[1]
dataset_path = os.path.join(folder_path, protein_name + '.csv')
protein_path = os.path.join(folder_path, protein_name + '.pdb')
Seed_everything(seed=42)

remove_water_path = os.path.join(folder_path,'remove_water')
if not os.path.exists(remove_water_path):
    os.makedirs(remove_water_path)

remove_water(protein_path, os.path.join(remove_water_path,f'{protein_name}.pdb'))
receptor_path = os.path.join(remove_water_path,f'{protein_name}.pdbqt')
convert_pdb_to_pdbqt(os.path.join(remove_water_path,f'{protein_name}.pdb'),receptor_path)
protein_remove_water = os.path.join(remove_water_path,f'{protein_name}.pdb')


def pocket_position_to_coordinate(path_pocket):
    pdb = pdbreader.read_pdb(path_pocket)
    pocket_name = os.path.split(path_pocket)[1].replace('_atm.pdb', '')
    residue_lst = " ".join(list(set([str(x) + ":" + str(a) for a, x in zip(pdb['ATOM']['resid'], pdb['ATOM']['chain'])])))
    mean_pocket = np.mean(np.concatenate([np.array(pdb['ATOM'][dim_name]).reshape(-1,1) for dim_name in ['x','y','z']], axis = 1), axis = 0)
    generation_path = os.path.join(folder_path, 'generation')
    if not os.path.exists(generation_path):
        os.makedirs(generation_path)

    pocket_path = os.path.join(generation_path, f'{pocket_name}.sdf')
    if not os.path.exists(pocket_path):
        cmd = f"python diffusion_generate/generate_ligands.py diffusion_generate/checkpoints/crossdocked_fullatom_cond.ckpt --pdbfile {protein_path} --outfile {pocket_path} --resi_list {residue_lst} --n_samples 10 --device {device}"
        os.system(cmd)
    return mean_pocket, pocket_path

def mol_to_list_smiles(mol_file):
    return sdf_to_smiles(mol_file)
    
mean_pockets_x = []
mean_pockets_y = []
mean_pockets_z = []

smile_lists   = []
key_idxs      = []
idx           = 0

if len(glob(f'{folder_path}/remove_water/{protein_name}_out/pockets/*.pdb')) == 0:
    cmd = f"fpocket -f {protein_remove_water}"
    os.system(cmd)

for path_pocket in glob(f'{folder_path}/remove_water/{protein_name}_out/pockets/*.pdb'):
    mean_pocket, pocket_path = pocket_position_to_coordinate(path_pocket)
    smile_list = mol_to_list_smiles(pocket_path)
    for i in range(len(smile_list)):
        mean_pockets_x.append(round(mean_pocket[0],2))
        mean_pockets_y.append(round(mean_pocket[1],2))
        mean_pockets_z.append(round(mean_pocket[2],2))
        smile_lists.append(smile_list[i])
        key_idxs.append(idx)
        data = {
                'key_idx'     : key_idxs,
                'smiles' : smile_lists,
                'x': mean_pockets_x,
                'y': mean_pockets_y,
                'z': mean_pockets_z,
                
            }
        idx += 1
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(os.path.split(pocket_path)[0], f'generation.csv'), index = False)

generation_path = os.path.join(os.path.split(pocket_path)[0], f'generation.csv')

def docking(generation_path, receptor_path):
    outfile_path = os.path.join(folder_path, 'generation_docking', 'generation_docking.csv')
    if os.path.exists(outfile_path):
        return outfile_path
    df = pd.read_csv(generation_path)
    df['ba'] = np.zeros(len(df))
    last_x, last_y, last_z = -99,-99,-99
    for i in range(len(df)):
        row = df.iloc[i]
        smile = row['smiles']
        x,y,z = row['x'], row['y'], row['z']
        if (x != last_x or y != last_y or z != last_z):
            func = Vina_3d(
                receptor_path,
                [float(x), float(y), float(z)],
                [40.0, 40.0, 40.0],
            )
            last_x, last_y, last_z = x,y,z
        smile = Chem.MolToSmiles(Chem.MolFromSmiles(smile), True)
        LigPrepper.smiles2pdbqt(smile, labels=f"{protein_name}_{row['key_idx']}")
        if not os.path.exists(os.path.join(folder_path, 'generation_docking')):
            os.makedirs(os.path.join(folder_path, 'generation_docking'))
        ba_generation = func(f"{protein_name}_{row['key_idx']}.pdbqt", os.path.join(folder_path, 'generation_docking',f"{row['key_idx']}.pdbqt"), n_poses = 5)
        df.loc[i, 'ba'] = ba_generation
        os.remove(f"{protein_name}_{row['key_idx']}.pdbqt")
        df.to_csv(outfile_path, index = False)
    return outfile_path
 
path_generate_docking = docking(generation_path, receptor_path)

def preprocessed_data(path_generate_ligand):
    out_file = os.path.join(folder_path, 'preprocessed_data', 'preprocessed_data.csv')
    if not os.path.exists(os.path.join(folder_path, 'preprocessed_data')):
        os.makedirs(os.path.join(folder_path, 'preprocessed_data'))
    if os.path.exists(out_file):
        return out_file
    
    gconv = GConv(
        input_dim=9, hidden_dim=64, activation=torch.nn.ReLU, num_layers=3
    ).to(device)
    fc1 = FC(hidden_dim=64 * 3)
    fc2 = FC(hidden_dim=64 * 3)
    encoder_model = Encoder(encoder=gconv, local_fc=fc1, global_fc=fc2).to(device)
    encoder_model.load_state_dict(torch.load(f"search_dgi/encoder_best.pt"))
    encoder_model.eval()

    generate = pd.read_csv(path_generate_ligand, encoding="utf-8")
    dataset = pd.read_csv(dataset_path, delimiter=",")

    docking_generation = []
    for i in range(len(generate["smiles"])):
        docking_generation.append(get_lig_graph(generate["smiles"][i]))


    dataloader = DataLoader(docking_generation, batch_size=512, shuffle=False)
    docking_generation_embed_lst = embed_data(encoder_model, dataloader, device)

    datasets = []
    for i in range(len(dataset["smiles"])):
        datasets.append(get_lig_graph(dataset["smiles"][i]))

    dataloader = DataLoader(datasets, batch_size=512, shuffle=False)
    datasets_embed_lst = embed_data(encoder_model, dataloader, device)
    
    docking_generation_embed = torch.stack(docking_generation_embed_lst).cpu().detach().numpy()
    datasets_embed           = torch.stack(datasets_embed_lst).cpu().detach().numpy()

    gnns_sim = cosine_similarity(datasets_embed, docking_generation_embed)
    potential_ligand_ba = list(generate["ba"])
    gnns_x, gnns_y, gnns_z = list(generate["x"]), list(generate["y"]), list(generate["z"])
    argmax_index = np.argmax(gnns_sim, axis=1)
    gnns_array = np.hstack(
        [
            gnns_sim[np.arange(0, argmax_index.shape[0]), argmax_index].reshape(-1, 1),
            np.array(potential_ligand_ba)[argmax_index].reshape(-1, 1),
            np.array(gnns_x)[argmax_index].reshape(-1, 1), 
            np.array(gnns_y)[argmax_index].reshape(-1, 1), 
            np.array(gnns_z)[argmax_index].reshape(-1, 1),
        ]
    )

    dataset["gnns_score"] = gnns_array[:, 0]
    dataset["gnns_ba"] = gnns_array[:, 1]
    dataset["x"] = gnns_array[:, 2]
    dataset["y"] = gnns_array[:, 3]
    dataset["z"] = gnns_array[:, 4]

    dataset["key_idx"] = dataset.index
    
    data_sort = dataset.sort_values(by = ['gnns_ba','gnns_score'], ascending=[True, False])
    data_sort['index'] = range(len(data_sort.sort_values(by = ['gnns_ba','gnns_score'])))
    data_sort.to_csv(out_file, index=False)
    return out_file

preprocessed_data_path = preprocessed_data(path_generate_docking)

def binding_data_creation(preprocessed_data_path, k = 100):
    out_file = os.path.join(folder_path, 'binding_data', 'binding_data.csv')
    if not os.path.exists(os.path.join(folder_path, 'binding_data')):
        os.makedirs(os.path.join(folder_path, 'binding_data'))
    if os.path.exists(out_file):
        return out_file


    df = pd.read_csv(preprocessed_data_path)
    df['ba'] = np.zeros(len(df))
    last_x, last_y, last_z = -99,-99,-99
    for i in range(k):
        row = df.iloc[i]
        smile = row['smiles']
        x,y,z = row['x'], row['y'], row['z']
        print(x,y,z)
        if (x != last_x or y != last_y or z != last_z):
            func = Vina_3d(
                receptor_path,
                [float(x), float(y), float(z)],
                [40.0, 40.0, 40.0],
            )
            last_x, last_y, last_z = x,y,z
        smile = Chem.MolToSmiles(Chem.MolFromSmiles(smile), True)
        LigPrepper.smiles2pdbqt(smile, labels=f"{protein_name}_{row['index']}")
        if not os.path.exists(os.path.join(folder_path, 'binding_data')):
            os.makedirs(os.path.join(folder_path, 'binding_data'))
        ba_generation = func(f"{protein_name}_{row['index']}.pdbqt", os.path.join(folder_path, 'binding_data',f"{row['index']}.pdbqt"), n_poses = 5)
        df.loc[i, 'ba'] = ba_generation
        os.remove(f"{protein_name}_{row['index']}.pdbqt")
        df.to_csv(out_file, index = False)
    return out_file

binding_data_creation(preprocessed_data_path)


args = arg_parsing_fix()
args.ckpt_path    = 'fabind/ckpt/best_model.bin'
args.resultFolder = '/cm/archive/phucpht/Drug_research/Blind_pocket/results'

lst_data = get_simulate_data(args = args, protein_path=receptor_path , dataset_path = dataset_path, group='train')
exit()
def database_distance_marking(optimal_weight_path, protein_name, dataset_name):
    ### Uncomment 
    if os.path.exists(f'/cm/shared/{user_name}/Molecule_Generation/Drug_repurposing/DrugBank/inference_data/{protein_name}_{dataset_name}_distance_lst.pt'):
        distance_path = f'/cm/shared/{user_name}/Molecule_Generation/Drug_repurposing/DrugBank/inference_data/{protein_name}_{dataset_name}_distance_lst.pt'
        index_path    = f'/cm/shared/{user_name}/Molecule_Generation/Drug_repurposing/DrugBank/inference_data/{protein_name}_{dataset_name}_index_lst.pt'
        return distance_path, index_path
    args = arg_parsing_inference()
    args.compound_coords_init_mode = "redocking"
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], mixed_precision=args.mixed_precision)
    pre = "test_inference"

    torch.multiprocessing.set_sharing_strategy('file_system')
    dataset = get_inference_data(protein_name, dataset_name)
    print(len(dataset))

    num_workers = 0
    data_loader = DataLoader(dataset, batch_size=args.batch_size, follow_batch=['x'], shuffle=False, pin_memory=False, num_workers=num_workers)

    model = get_model(args, None)
    model = accelerator.prepare(model)
    args.ckpt = optimal_weight_path
    model.load_state_dict(torch.load(args.ckpt))
    model.to(device)

    set_seed(args.seed)
    model.eval()

    data_iter = tqdm(data_loader, mininterval=args.tqdm_interval, disable=not accelerator.is_main_process)
    distance_path =  f'/cm/shared/{user_name}/Molecule_Generation/Drug_repurposing/DrugBank/inference_data/{protein_name}_{dataset_name}_distance_lst.pt'
    index_path    =  f'/cm/shared/{user_name}/Molecule_Generation/Drug_repurposing/DrugBank/inference_data/{protein_name}_{dataset_name}_index_lst.pt'
    distance_lst = []
    index_lst    = []
    for batch_id, data in enumerate(data_iter):
        try:
            data = data.to(device)
            _, _, dis_to_pro_lst, index = model.inference(data) 
            distance_lst.append([dis.detach().cpu() for dis in dis_to_pro_lst])
            # print(distance_lst)
            index_lst.append(index.detach().cpu())
        except:
            continue

        distance_torch = [torch.tensor(inner_list) for inner_list in distance_lst]
        distance_torch = torch.concat(distance_torch).flatten()
        index_torch    = torch.concat(index_lst).flatten()
        torch.save(distance_torch, distance_path)
        torch.save(index_torch, index_path)
        torch.cuda.empty_cache()

    if os.path.exists(f"/cm/shared/{user_name}/Molecule_Generation/Drug_repurposing/DrugBank/ckpt/"):
        shutil.rmtree(f"/cm/shared/{user_name}/Molecule_Generation/Drug_repurposing/DrugBank/ckpt/")
    return distance_path, index_path

def create_distance_dataset(protein_name, dataset_name, distance_path, index_path):
    drugbank_df = pd.read_csv(f'/cm/shared/{user_name}/Molecule_Generation/Drug_repurposing/DrugBank/data/{dataset_name}.csv')
    drugbank_df["key_idx"] = drugbank_df.index
    drugbank_df['distance'] = 1e3
    drugbank_df.loc[torch.load(index_path).detach().cpu().numpy(), 'distance'] = torch.load(distance_path).detach().cpu().numpy()
    df_sorted_by_dis = drugbank_df.sort_values(by='distance')
    if not os.path.exists(f'/cm/shared/{user_name}/Molecule_Generation/Drug_repurposing/DrugBank/distance_data/'):
        os.makedirs(f'/cm/shared/{user_name}/Molecule_Generation/Drug_repurposing/DrugBank/distance_data/')
    df_sorted_by_dis.to_csv(f'/cm/shared/{user_name}/Molecule_Generation/Drug_repurposing/DrugBank/distance_data/distance_{protein_name}_{dataset_name}.csv',index=False)
