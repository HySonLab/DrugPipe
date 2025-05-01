# 1. Splits 1/10 data                           DONE
# 2. Download all proteins in the subset        DONE
# 3. Change protein into pdbqt                  DONE
# 4. Change drug into pdbqt                     DONE
# 5. Docking protein and drug by qvina-w        DONE

from glob import glob
import os
import pandas as pd
import time
protein_paths = sorted(glob('/home/phuc.phamhuythienai@gmail.com/Desktop/New/DrugPipe/datasets/case_study/*pdb'))
drugbank_conformers = sorted(glob('/home/phuc.phamhuythienai@gmail.com/Desktop/New/DrugPipe/datasets/drugbank_conformation/*sdf'))

def convert_pdb_to_pdbqt(pdb_file, pdbqt_file):
    cmd = f"obabel {pdb_file} -O {pdbqt_file} -h -gen3D -xr"
    # Using os.system() method
    os.system(cmd)
    
def convert_sdf_to_pdbqt(sdf_file, pdbqt_file):
    cmd = f"obabel {sdf_file} -O {pdbqt_file}"
    # Using os.system() method
    os.system(cmd)
    
def parse_pdbqt_coordinates(pdbqt_path):
    x_coords, y_coords, z_coords = [], [], []

    with open(pdbqt_path, 'r') as file:
        for line in file:
            # print(line)
            if line.startswith("ATOM") or line.startswith("HETATM"):
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                x_coords.append(x)
                y_coords.append(y)
                z_coords.append(z)

    if not x_coords:
        raise ValueError("No atom coordinates found in the file.")

    # Center (mean)
    center_x = sum(x_coords) / len(x_coords)
    center_y = sum(y_coords) / len(y_coords)
    center_z = sum(z_coords) / len(z_coords)

    # Size (range)
    size_x = max(x_coords) - min(x_coords)
    size_y = max(y_coords) - min(y_coords)
    size_z = max(z_coords) - min(z_coords)

    return {
        "center_x": center_x,
        "center_y": center_y,
        "center_z": center_z,
        "size_x": size_x,
        "size_y": size_y,
        "size_z": size_z
    }
    
def extract_affinities_from_qvina_output(filepath):
    affinities = []
    in_result_section = False

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("mode |"):
                in_result_section = True
                continue
            if in_result_section and line and line[0].isdigit():
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        affinity = float(parts[1])
                        affinities.append(affinity)
                    except ValueError:
                        continue

    if not affinities:
        raise ValueError("No affinities found in the file.")

    mean_affinity = sum(affinities) / len(affinities)
    return mean_affinity, affinities

"""
# PBP1a     - 1B2Y
# HPA       - 4OON
# COVID19   - 7B3B
# HIV1      - 2JLE
"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--protein_index', type=int)
args = parser.parse_args()

protein_path = protein_paths[args.protein_index]

start_0 = time.time()
protein_name = os.path.split(protein_path)[1].replace('.pdb', '')
aggregation_df = pd.read_csv(f'/home/phuc.phamhuythienai@gmail.com/Desktop/New/qvina_result/join_{protein_name}.csv')
lst_docked_ligand = list(aggregation_df['Ligand_ID'])

protein_pdbqt_path = protein_path.replace('pdb', 'pdbqt')
convert_pdb_to_pdbqt(protein_path, protein_pdbqt_path)
box_dict = parse_pdbqt_coordinates(protein_pdbqt_path)
center_x, center_y, center_z, size_x, size_y, size_z = box_dict['center_x'], box_dict['center_y'], box_dict['center_z'], box_dict['size_x'], box_dict['size_y'], box_dict['size_z']
results_df = pd.DataFrame(columns=["Ligand_ID", "Mean_Affinity", "Time"])


for idex, sdf_path in enumerate(drugbank_conformers):
    start = time.time()
    ligand_name = os.path.split(sdf_path)[1].replace('.sdf', '')
    ligand_pdbqt_path = sdf_path.replace('sdf', 'pdbqt')
    if not os.path.exists(ligand_pdbqt_path):
        convert_sdf_to_pdbqt(sdf_path, ligand_pdbqt_path)
    cmd = f"./qvina-w --receptor {protein_pdbqt_path} --ligand {ligand_pdbqt_path} --center_x {center_x} --center_y {center_y} --center_z {center_z} --size_x {size_x} --size_y {size_y} --size_z {size_z} --log qvina_result/aa.txt --out {protein_name}_{args.ligand_index}_{args.protein_index}.pqbqt"
    # Using os.system() method
    os.system(cmd)
    try:
        output_file = "qvina_result/test.txt" 
        mean_affinity, all_affinities = extract_affinities_from_qvina_output(output_file)
        end = time.time()
        results_df.loc[len(results_df)] = [ligand_name, round(mean_affinity, 2), end - start]
        print(f"✅ {protein_name}-{ligand_name}: {mean_affinity:.2f} kcal/mol")
    except Exception as e:
        print(f"❌ Failed for {ligand_name}: {e}")

    # Save the DataFrame to CSV
    output_csv = f"qvina_result/{protein_name}_{args.protein_index}_{args.ligand_index}.csv"
    results_df_sorted = results_df.sort_values(by='Mean_Affinity', ascending=True)  # or True for ascending
    results_df_sorted.to_csv(output_csv, index=False)
    print(f"📁 Saved results to {output_csv}")
end_0 = time.time()
print('Time: ', end_0 - start_0)