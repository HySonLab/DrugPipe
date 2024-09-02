import os
import sys
from utils_pair import *
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
from torch_geometric.data import Batch
import warnings
import copy
from tdc.chem_utils import MolConvert
warnings.filterwarnings("ignore")
Seed_everything(seed=42)


v = Vina(sf_name='vina')
folder = "case_study"
user_name = 'phucpht'
# ====== Global variables ======
root = f"/cm/shared/{user_name}/Drug_research/Blind_pocket/data/"
path_pdbs = sorted(glob(os.path.join(root, f"{folder}/pdb/*.pdb")))

# name_dataset = "drugbank"
# dataset_path = f"/cm/shared/{user_name}/Drug_research/Blind_pocket/DrugBank/data/{name_dataset}.csv"
device = torch.device("cuda:5")
idx = int(sys.argv[1])
path = path_pdbs[idx]


# =================================
def blind_ligand_generation(pdb_path):
    protein_name = os.path.split(pdb_path)[1]
    target_name = protein_name.replace(".pdb", "")
    csv_path = os.path.join(root, f'{folder}/ligand_generation/{target_name}.csv')
    if not os.path.exists(csv_path):
        os.system(f'''cd ./vae_generation/
            python generate_with_specific_target.py --root {root} --folder {folder} --protein_name {protein_name} --device {device}''')
    
    generate_df = pd.read_csv(csv_path)

    path_pdbqt = pdb_path.replace(".pdb", ".pdbqt")
    prepare_protein_pdbqt(pdb_path,path_pdbqt)

    protein_df = fo.bp_to_df(fo.read_pdb(path_pdbqt))
    center_x, center_y, center_z = (
        np.mean(protein_df["x"]),
        np.mean(protein_df["y"]),
        np.mean(protein_df["z"]),
    )

    func = Vina_3d(
        path_pdbqt,
        [float(center_x), float(center_y), float(center_z)],
        [70.0, 70.0, 70.0],
    )
    for i in range(len(generate_df)):
        try:
            row = generate_df.iloc[i]
            smiles = row['smiles']
            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), True)
            out_ligand_folder = f"/cm/shared/phucpht/Drug_research/Blind_pocket/data/{folder}/{target_name}_binding_ligands_stage_1"
            name_out_ligands = str(row["key_idx"])
            out_ligand = create_and_merge_path(
                out_ligand_folder, f"{str(name_out_ligands)}.sdf"
            )
            create_ligand(smiles, out_ligand)
            command = ['mk_prepare_ligand.py', '-i', out_ligand, '-o', out_ligand.replace("sdf","pdbqt")]
            subprocess.run(command)
            pdbqt_file = out_ligand.replace("sdf","pdbqt")
            ba_generation = func(pdbqt_file, out_ligand)
            generate_df.at[i, 'ba'] = ba_generation
        except:
            continue
        generate_df.to_csv(csv_path, index = False)
    return csv_path


blind_ligand_generation(path)
exit()
def preprocessed_data(path_generate_ligand):
    protein_name = os.path.split(path_generate_ligand)[1].split("_")[0]
    out_folder = f"/cm/shared/{user_name}/Molecule_Generation/Drug_repurposing/DrugBank/preprocessed_data"
    file_name = f"preprocessed_{protein_name}_{name_dataset}_data.csv"
    out_file = create_and_merge_path(out_folder, file_name)
    if os.path.exists(out_file):
        return out_file
    
    gconv = GConv(
        input_dim=9, hidden_dim=64, activation=torch.nn.ReLU, num_layers=3
    ).to(device)
    fc1 = FC(hidden_dim=64 * 3)
    fc2 = FC(hidden_dim=64 * 3)
    encoder_model = Encoder(encoder=gconv, local_fc=fc1, global_fc=fc2).to(device)
    encoder_model.load_state_dict(torch.load(f"/cm/shared/{user_name}/Molecule_Generation/Drug_repurposing/DrugBank/search_dgi/encoder_best.pt"))
    encoder_model.eval()
    
    out_folder = f"/cm/shared/{user_name}/Molecule_Generation/Drug_repurposing/DrugBank/preprocessed_data"
    file_name = f"preprocessed_{protein_name}_{name_dataset}_data.csv"
    out_file = create_and_merge_path(out_folder, file_name)
    if os.path.exists(out_file):
        return out_file
    generate = pd.read_csv(path_generate_ligand, encoding="utf-8")
    drug_bank = pd.read_csv(dataset_path, delimiter=",")

    tanimoto_array = []
    morgan_array = []

    for i in tqdm(range(len(drug_bank["smiles"]))):
        max_sim_tanimoto = 0
        min_ba_tanimoto = 0

        max_sim_morgan = 0
        min_ba_morgan = 0

        for j in range(len(generate["smiles"])):
            try:
                # tanimoto
                tanimoto = similarity(drug_bank["smiles"][i], generate["smiles"][j])
                if tanimoto > max_sim_tanimoto:
                    max_sim_tanimoto = tanimoto
                    min_ba_tanimoto = generate["BA"][j]

                # morgan
                fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2)
                m1 = Chem.MolFromSmiles(drug_bank["smiles"][i])
                m2 = Chem.MolFromSmiles(generate["smiles"][j])
                fp1 = fpgen.GetSparseCountFingerprint(m1)
                fp2 = fpgen.GetSparseCountFingerprint(m2)
                morgan_sim = DataStructs.DiceSimilarity(fp1, fp2)
                if morgan_sim > max_sim_morgan:
                    max_sim_morgan = morgan_sim
                    min_ba_morgan = generate["BA"][j]
            except:
                continue
        tanimoto_array.append([max_sim_tanimoto, min_ba_tanimoto])
        morgan_array.append([max_sim_morgan, min_ba_morgan])
    tanimoto_array = np.array(tanimoto_array)
    morgan_array = np.array(morgan_array)
    data = []
    for i in range(len(generate["smiles"])):
        data.append(get_lig_graph(generate["smiles"][i]))
    wrong_smiles = pickle.load(
        open(f"/cm/shared/{user_name}/Molecule_Generation/Drug_repurposing/DrugBank/graph_data/wrong_smiles_{name_dataset}.pkl", "rb")
    )
    right_smiles = pickle.load(
        open(f"/cm/shared/{user_name}/Molecule_Generation/Drug_repurposing/DrugBank/graph_data/right_smiles_{name_dataset}.pkl", "rb")
    )

    dataloader = DataLoader(data, batch_size=512, shuffle=False)
    embed_lst = embed_data(encoder_model, dataloader, device)

    if not os.path.exists(f"/cm/shared/{user_name}/Molecule_Generation/Drug_repurposing/DrugBank/graph_data/{name_dataset}_embed.pkl"):
        create_embedding(name_file = f'{name_dataset}', device=device)

    drug_bank_embed = torch.stack(
        pickle.load(open(f"/cm/shared/{user_name}/Molecule_Generation/Drug_repurposing/DrugBank/graph_data/{name_dataset}_embed.pkl", "rb"))
    )
    embedding_dim = drug_bank_embed.shape[1]

    potential_embed = torch.stack(embed_lst)

    num_potential = potential_embed.shape[0]
    drug_bank_mask_error = torch.zeros(
        (len(wrong_smiles) + len(right_smiles), embedding_dim)
    )
    drug_bank_mask_error[right_smiles] = drug_bank_embed.detach().cpu()

    potential_embed = potential_embed.detach().cpu().numpy()
    drug_bank_mask_error = drug_bank_mask_error.detach().cpu().numpy()
    gnns_sim = cosine_similarity(drug_bank_mask_error, potential_embed)
    infinity_embed = np.ones((len(wrong_smiles), num_potential)) * -1
    gnns_sim[wrong_smiles] = infinity_embed
    potential_ligand_ba = list(generate["BA"])
    argmax_index = np.argmax(gnns_sim, axis=1)
    gnns_array = np.hstack(
        [
            gnns_sim[np.arange(0, argmax_index.shape[0]), argmax_index].reshape(-1, 1),
            np.array(potential_ligand_ba)[argmax_index].reshape(-1, 1),
        ]
    )
    drug_bank["tanimoto_score"] = tanimoto_array[:, 0]
    drug_bank["tanimoto_ba"] = tanimoto_array[:, 1]
    drug_bank["morgan_score"] = morgan_array[:, 0]
    drug_bank["morgan_ba"] = morgan_array[:, 1]
    drug_bank["gnns_score"] = gnns_array[:, 0]
    drug_bank["gnns_ba"] = gnns_array[:, 1]
    drug_bank["key_idx"] = drug_bank.index
    drug_bank.to_csv(out_file, index=False)
    print("save ", file_name)
    return out_file

def binding_data_creation(preprocessed_data_path, path_pdbqt=path_pdbs):
    
    protein_name = os.path.split(preprocessed_data_path)[1].split("_")[1]
    dataset_name = name_dataset
    file_name = f"binding_data_{protein_name}_{name_dataset}.csv"
    out_folder = f"/cm/shared/{user_name}/Molecule_Generation/Drug_repurposing/DrugBank/binding_data"
    out_file = create_and_merge_path(out_folder, file_name)
    if os.path.exists(out_file):
        out_ligand_folder = f"/cm/shared/{user_name}/Molecule_Generation/Drug_repurposing/DrugBank/docking_output/{protein_name}_{dataset_name}_binding_ligands_stage_1"
        return out_file, out_ligand_folder  # out_folder_ligand
    database_sim = pd.read_csv(preprocessed_data_path)
    mask = paretoset(
        database_sim[
            [
                "tanimoto_score",
                "tanimoto_ba",
                "morgan_score",
                "morgan_ba",
                "gnns_score",
                "gnns_ba",
            ]
        ],
        sense=["max", "min", "max", "min", "max", "min"],
    )
    multi_objective_ranking_frame = database_sim[mask]
    receptor_file_path = os.path.join(path_pdbqt, protein_name + ".pdbqt")
    protein_df = fo.bp_to_df(fo.read_pdb(receptor_file_path))
    
    center_x, center_y, center_z = (
        np.mean(protein_df["x"]),
        np.mean(protein_df["y"]),
        np.mean(protein_df["z"]),
    )
    v.set_receptor(receptor_file_path)
    v.compute_vina_maps(center=[float(center_x), float(center_y), float(center_z)], box_size=[35.0, 35.0, 35.0])

    print(multi_objective_ranking_frame)
    ba_lst = np.ones(len(multi_objective_ranking_frame["smiles"]))
    for idx, smiles in enumerate(multi_objective_ranking_frame["smiles"]):
        try:
            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), True)
            out_ligand_folder = f"/cm/shared/{user_name}/Molecule_Generation/Drug_repurposing/DrugBank/docking_output/{protein_name}_{dataset_name}_binding_ligands_stage_1"
            name_out_ligands = str(multi_objective_ranking_frame["key_idx"].iloc[idx])
            out_ligand = create_and_merge_path(
                out_ligand_folder, f"{str(name_out_ligands)}.sdf"
            )
            create_ligand(smiles, out_ligand)
            command = ['mk_prepare_ligand.py', '-i', out_ligand, '-o', out_ligand.replace("sdf","pdbqt")]
            subprocess.run(command)

            v.set_ligand_from_file(out_ligand.replace("sdf","pdbqt"))
            v.dock(exhaustiveness = 4, n_poses=5)
            top_5_scores = list(v.energies(n_poses=5, energy_range = float('inf'))[:, 0])
            ba_lst[idx] = top_5_scores[0]
        except:
            ba_lst[idx] = np.inf
        multi_objective_ranking_frame["BA_docking_stage_1"] = ba_lst
    
    out_folder = f"/cm/shared/{user_name}/Molecule_Generation/Drug_repurposing/DrugBank/binding_data"
    file_name = f"binding_data_{protein_name}_{name_dataset}.csv"
    print("save ", file_name)
    out_file = create_and_merge_path(out_folder, file_name)
    multi_objective_ranking_frame.to_csv(out_file, index=False)
    return out_file, out_ligand_folder  # out_folder_ligand

def binding_fine_tune(out_ligand):
    protein_name = os.path.split(out_ligand)[1].split("_")[0]
    database_name = name_dataset

    stage = os.path.split(out_ligand)[1].split("_")[-1]
    
    weight_path = f'/cm/shared/{user_name}/Molecule_Generation/Drug_repurposing/DrugBank/ckpt'
    all_finetune_weights = sorted(glob(f'{weight_path}/*{protein_name}*'), key = lambda x: float(os.path.split(x)[1].split('_')[4][:-4]))
    # if len(all_finetune_weights) > 0:
    #     return all_finetune_weights[0]
    
    ligand_docking_out = glob(f"{out_ligand}/*.pdbqt")
    for ligand_path in ligand_docking_out:
        file_name = os.path.split(ligand_path)[1].split(".")[0] + ".pdb"
        out_folder = os.path.join(
            f"/cm/shared/{user_name}/Molecule_Generation/Drug_repurposing/DrugBank/ligands_conformation/stage_{stage}", f"{protein_name}_{database_name}_binding_ligands"
        )
        outfile = create_and_merge_path(out_folder, file_name)
        cmd = f"obabel {ligand_path} -O {outfile}"
        os.system(cmd)

    protein_path = os.path.join(path_pdbs, f"{protein_name}.pdbqt")
    out_protein = f"/cm/shared/{user_name}/Molecule_Generation/Drug_repurposing/DrugBank/ligands_conformation/stage_{stage}/{protein_name}.pdb"
    cmd = f"obabel {protein_path} -O {out_protein}"
    os.system(cmd)
    args = arg_parsing_fix()
    lst_data = get_simulate_data(args = args, protein_name=protein_name, database=database_name, group='train')
    data_loader = DataLoader(
        lst_data,
        batch_size=args.batch_size,
        follow_batch=["x", "compound_pair"],
        shuffle=True,
        pin_memory=False,
        num_workers=10,
    )


    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs], mixed_precision=args.mixed_precision
    )
    set_seed(args.seed)
    model = get_model(args, None)
    model = accelerator.prepare(model)
    model.load_state_dict(torch.load(f"/cm/shared/{user_name}/Molecule_Generation/Drug_repurposing/DrugBank/fabind/ckpt/best_model.bin"))
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    last_epoch = -1
    scheduler_warm_up = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.5,
        end_factor=1,
        total_iters=args.warmup_epochs * len(data_loader),
        last_epoch=last_epoch,
    )
    scheduler_post = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=(args.total_epochs - args.warmup_epochs) * len(data_loader),
        last_epoch=last_epoch,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[scheduler_warm_up, scheduler_post],
        milestones=[args.warmup_epochs * len(data_loader)],
    )
    (
        model,
        optimizer,
        scheduler,
        data_loader,
    ) = accelerator.prepare(
        model,
        optimizer,
        scheduler,
        data_loader,
    )

    criterion = nn.MSELoss()
    com_coord_criterion = nn.SmoothL1Loss()
    pocket_cls_criterion = nn.BCEWithLogitsLoss(reduction="mean")

    pocket_coord_criterion = nn.HuberLoss(delta=args.pocket_coord_huber_delta)


    for epoch in range(last_epoch + 1, args.total_epochs):
    # for epoch in range(0, 1):
        model.train()

        y_list = []
        y_pred_list = []
        com_coord_list = []
        com_coord_pred_list = []
        rmsd_list = []
        rmsd_2A_list = []
        rmsd_5A_list = []
        centroid_dis_list = []
        centroid_dis_2A_list = []
        centroid_dis_5A_list = []
        pocket_coord_list = []
        pocket_coord_pred_list = []
        # pocket_coord_pred_for_update_list = []
        pocket_cls_list = []
        pocket_cls_pred_list = []
        pocket_cls_pred_round_list = []
        protein_len_list = []
        # pdb_list = []
        count = 0
        skip_count = 0
        batch_loss = 0.0
        batch_by_pred_loss = 0.0
        batch_distill_loss = 0.0
        com_coord_batch_loss = 0.0
        pocket_cls_batch_loss = 0.0
        pocket_coord_batch_loss = 0.0
        keepNode_less_5_count = 0

        if args.disable_tqdm:
            data_iter = data_loader
        else:
            data_iter = tqdm(
                data_loader,
                mininterval=args.tqdm_interval,
                disable=not accelerator.is_main_process,
            )
        for batch_id, data in enumerate(data_iter, start=1):
            torch.cuda.empty_cache()
            model.to(device)
            data = data.to(device)
            optimizer.zero_grad()
            # Denote num_atom as N, num_amino_acid_of_pocket as M, num_amino_acid_of_protein as L
            # com_coord_pred: [B x N, 3]
            # y_pred, y_pred_by_coord: [B, N x M]
            # pocket_cls_pred, protein_out_mask_whole: [B, L]
            # p_coords_batched_whole: [B, L, 3]
            # pred_pocket_center: [B, 3]

            try:
                (
                    com_coord_pred,
                    compound_batch,
                    y_pred,
                    y_pred_by_coord,
                    pocket_cls_pred,
                    pocket_cls,
                    protein_out_mask_whole,
                    p_coords_batched_whole,
                    pred_pocket_center,
                    dis_map,
                    keepNode_less_5,
                ) = model(data, train=True)
            except:
                continue
            # y = data.y 
            if (
                y_pred.isnan().any()
                or com_coord_pred.isnan().any()
                or pocket_cls_pred.isnan().any()
                or pred_pocket_center.isnan().any()
                or y_pred_by_coord.isnan().any()
            ):
                print(f"nan occurs in epoch {epoch}")
                continue
            com_coord = data.coords
            pocket_cls_loss = (
                args.pocket_cls_loss_weight
                * pocket_cls_criterion(pocket_cls_pred, pocket_cls.float())
                * (protein_out_mask_whole.numel() / protein_out_mask_whole.sum())
            )
            pocket_coord_loss = args.pocket_distance_loss_weight * pocket_coord_criterion(
                pred_pocket_center, data.coords_center
            )
            contact_loss = (
                args.pair_distance_loss_weight * criterion(y_pred, dis_map)
                if len(dis_map) > 0
                else torch.tensor([0])
            )
            contact_by_pred_loss = (
                args.pair_distance_loss_weight * criterion(y_pred_by_coord, dis_map)
                if len(dis_map) > 0
                else torch.tensor([0])
            )
            contact_distill_loss = (
                args.pair_distance_distill_loss_weight * criterion(y_pred_by_coord, y_pred)
                if len(y_pred) > 0
                else torch.tensor([0])
            )

            com_coord_loss = (
                args.coord_loss_weight * com_coord_criterion(com_coord_pred, com_coord)
                if len(com_coord) > 0
                else torch.tensor([0])
            )

            sd = ((com_coord_pred.detach() - com_coord) ** 2).sum(dim=-1)
            rmsd = scatter_mean(sd, index=compound_batch, dim=0).sqrt().detach()

            centroid_pred = scatter_mean(src=com_coord_pred, index=compound_batch, dim=0)
            centroid_true = scatter_mean(src=com_coord, index=compound_batch, dim=0)
            centroid_dis = (centroid_pred - centroid_true).norm(dim=-1)

            loss = (
                com_coord_loss
                + contact_loss
                + contact_by_pred_loss
                + contact_distill_loss
                + pocket_cls_loss
                + pocket_coord_loss
            )

            accelerator.backward(loss)
            if args.clip_grad:
                # clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            batch_loss += len(y_pred) * contact_loss.item()
            batch_by_pred_loss += len(y_pred_by_coord) * contact_by_pred_loss.item()
            batch_distill_loss += len(y_pred_by_coord) * contact_distill_loss.item()
            com_coord_batch_loss += len(com_coord_pred) * com_coord_loss.item()

            pocket_cls_batch_loss += len(pocket_cls_pred) * pocket_cls_loss.item()
            pocket_coord_batch_loss += len(pred_pocket_center) * pocket_coord_loss.item()

            keepNode_less_5_count += keepNode_less_5

            y_list.append(dis_map.detach())
            y_pred_list.append(y_pred.detach())
            com_coord_list.append(com_coord)
            com_coord_pred_list.append(com_coord_pred.detach())
            rmsd_list.append(rmsd.detach())
            rmsd_2A_list.append((rmsd.detach() < 2).float())
            rmsd_5A_list.append((rmsd.detach() < 5).float())
            centroid_dis_list.append(centroid_dis.detach())
            centroid_dis_2A_list.append((centroid_dis.detach() < 2).float())
            centroid_dis_5A_list.append((centroid_dis.detach() < 5).float())

            batch_len = protein_out_mask_whole.sum(dim=1).detach()
            protein_len_list.append(batch_len)
            pocket_coord_pred_list.append(pred_pocket_center.detach())
            pocket_coord_list.append(data.coords_center)
            # use hard to calculate acc and skip samples
            for i, j in enumerate(batch_len):
                count += 1
                pocket_cls_list.append(pocket_cls.detach()[i][:j])
                pocket_cls_pred_list.append(pocket_cls_pred.detach()[i][:j].sigmoid())
                pocket_cls_pred_round_list.append(
                    pocket_cls_pred.detach()[i][:j].sigmoid().round().int()
                )
                pred_index_bool = (
                    pocket_cls_pred.detach()[i][:j].sigmoid().round().int() == 1
                )
                if pred_index_bool.sum() == 0:  # all the prediction is False, skip
                    skip_count += 1

            if batch_id % args.log_interval == 0:
                stats_dict = {}
                stats_dict["step"] = batch_id
                stats_dict["lr"] = optimizer.param_groups[0]["lr"]
                stats_dict["contact_loss"] = contact_loss.item()
                stats_dict["contact_by_pred_loss"] = contact_by_pred_loss.item()
                stats_dict["contact_distill_loss"] = contact_distill_loss.item()
                stats_dict["com_coord_loss"] = com_coord_loss.item()
                stats_dict["pocket_cls_loss"] = pocket_cls_loss.item()
                stats_dict["pocket_coord_loss"] = pocket_coord_loss.item()
                
        # y = torch.cat(y_list)
        y_pred = torch.cat(y_pred_list)
        # y, y_pred = accelerator.gather((y, y_pred))

        com_coord = torch.cat(com_coord_list)
        com_coord_pred = torch.cat(com_coord_pred_list)
        # com_coord, com_coord_pred = accelerator.gather((com_coord, com_coord_pred))

        rmsd = torch.cat(rmsd_list)
        rmsd_2A = torch.cat(rmsd_2A_list)
        rmsd_5A = torch.cat(rmsd_5A_list)
        # rmsd, rmsd_2A, rmsd_5A = accelerator.gather((rmsd, rmsd_2A, rmsd_5A))
        rmsd_25 = torch.quantile(rmsd, 0.25)
        rmsd_50 = torch.quantile(rmsd, 0.50)
        rmsd_75 = torch.quantile(rmsd, 0.75)

        centroid_dis = torch.cat(centroid_dis_list)
        centroid_dis_2A = torch.cat(centroid_dis_2A_list)
        centroid_dis_5A = torch.cat(centroid_dis_5A_list)
        # centroid_dis, centroid_dis_2A, centroid_dis_5A = accelerator.gather((centroid_dis, centroid_dis_2A, centroid_dis_5A))
        centroid_dis_25 = torch.quantile(centroid_dis, 0.25)
        centroid_dis_50 = torch.quantile(centroid_dis, 0.50)
        centroid_dis_75 = torch.quantile(centroid_dis, 0.75)

        pocket_cls = torch.cat(pocket_cls_list)
        pocket_cls_pred = torch.cat(pocket_cls_pred_list)
        pocket_cls_pred_round = torch.cat(pocket_cls_pred_round_list)
        pocket_coord_pred = torch.cat(pocket_coord_pred_list)
        pocket_coord = torch.cat(pocket_coord_list)
        protein_len = torch.cat(protein_len_list)

        pocket_cls_accuracy = (pocket_cls_pred_round == pocket_cls).sum().item() / len(
            pocket_cls_pred_round
        )

        metrics = {
            "samples": count,
            "skip_samples": skip_count,
            "keepNode < 5": keepNode_less_5_count,
        }
        metrics.update(
            {
                "contact_loss": batch_loss / len(y_pred),
                "contact_by_pred_loss": batch_by_pred_loss / len(y_pred),
                "contact_distill_loss": batch_distill_loss / len(y_pred),
            }
        )
        metrics.update({"com_coord_huber_loss": com_coord_batch_loss / len(com_coord_pred)})
        metrics.update(
            {
                "rmsd": rmsd.mean().item(),
                "rmsd < 2A": rmsd_2A.mean().item(),
                "rmsd < 5A": rmsd_5A.mean().item(),
            }
        )
        metrics.update(
            {
                "rmsd 25%": rmsd_25.item(),
                "rmsd 50%": rmsd_50.item(),
                "rmsd 75%": rmsd_75.item(),
            }
        )
        metrics.update(
            {
                "centroid_dis": centroid_dis.mean().item(),
                "centroid_dis < 2A": centroid_dis_2A.mean().item(),
                "centroid_dis < 5A": centroid_dis_5A.mean().item(),
            }
        )
        metrics.update(
            {
                "centroid_dis 25%": centroid_dis_25.item(),
                "centroid_dis 50%": centroid_dis_50.item(),
                "centroid_dis 75%": centroid_dis_75.item(),
            }
        )

        metrics.update(
            {"pocket_cls_bce_loss": pocket_cls_batch_loss / len(pocket_coord_pred)}
        )
        metrics.update(
            {"pocket_coord_mse_loss": pocket_coord_batch_loss / len(pocket_coord_pred)}
        )
        metrics.update({"pocket_cls_accuracy": pocket_cls_accuracy})
        metrics.update(pocket_metrics(pocket_coord_pred, pocket_coord))

        accelerator.wait_for_everyone()

        # metrics_list.append(metrics)
        # release memory
        y, y_pred = None, None
        com_coord, com_coord_pred = None, None
        rmsd, rmsd_2A, rmsd_5A = None, None, None
        centroid_dis, centroid_dis_2A, centroid_dis_5A = None, None, None
        (
            pocket_cls,
            pocket_cls_pred,
            pocket_cls_pred_round,
            pocket_coord_pred,
            pocket_coord,
            protein_len,
        ) = (None, None, None, None, None, None)
        use_y_mask = False

        if not os.path.exists(f"/cm/shared/{user_name}/Molecule_Generation/Drug_repurposing/DrugBank/ckpt/"):
            os.makedirs(f"/cm/shared/{user_name}/Molecule_Generation/Drug_repurposing/DrugBank/ckpt/")
            
        torch.save(model.state_dict(), f"/cm/shared/{user_name}/Molecule_Generation/Drug_repurposing/DrugBank/ckpt/finetune_fabind_{protein_name}_{database_name}_{round(metrics['com_coord_huber_loss'],4)}.pth")
        accelerator.wait_for_everyone()
    weight_path = f'/cm/shared/{user_name}/Molecule_Generation/Drug_repurposing/DrugBank/ckpt'
    all_finetune_weights = sorted(glob(f'{weight_path}/*{protein_name}*'), key = lambda x: float(os.path.split(x)[1].split('_')[4][:-4]))
    optimal_weight_path = all_finetune_weights[0]
    return optimal_weight_path

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

    





# preprocessed_data_path = preprocessed_data(path_generate_ligands[idx])

# protein_name = os.path.split(preprocessed_data_path)[1].split('_')[1]
# dataset_name = name_dataset
# out_file, out_ligand   = binding_data_creation(preprocessed_data_path)
# optimal_weight_path = binding_fine_tune(out_ligand)
# distance_path, index_path = database_distance_marking(optimal_weight_path, protein_name, dataset_name)
# create_distance_dataset(protein_name, dataset_name, distance_path, index_path)