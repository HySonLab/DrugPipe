from rdkit import Chem
from rdkit.Chem import Descriptors, QED
import sascorer  # Import file vừa tải
drugbank_data = pd.read_csv('generate_smile_cond_full_atoms.csv')
smiles_lst = list(drugbank_data["smiles"])
# Tạo dictionary chứa danh sách các thuộc tính
property_valid_lst = {
    "SMILES": [], "MW": [], "PSA": [], "HBD": [], "HBA": [], "Rotatable Bonds": [],
    "Aromatic Rings": [], "Heavy Atoms": [], "SA Score": [], "QED Score": []
}
smile_valid_lst = []  # Lưu danh sách SMILES hợp lệ
for smiles_string in smiles_lst:
    mol = Chem.MolFromSmiles(smiles_string)

    if mol is None:
        print(f"Invalid SMILES string: {smiles_string}")
    else:
        # Tính toán các thuộc tính phân tử
        mw = Descriptors.MolWt(mol)
        psa = Descriptors.TPSA(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        rot_bonds = Descriptors.NumRotatableBonds(mol)
        aromatic_rings = Descriptors.NumAromaticRings(mol)
        heavy_atoms = Descriptors.HeavyAtomCount(mol)
        sa_score = sascorer.calculateScore(mol)
        qed_score = QED.qed(mol)
        property_valid_lst['SMILES'].append(smiles_string)
        # Thêm danh sách các giá trị vào từng key trong dictionary
        property_valid_lst["MW"].append(mw)
        property_valid_lst["PSA"].append(psa)
        property_valid_lst["HBD"].append(hbd)
        property_valid_lst["HBA"].append(hba)
        property_valid_lst["Rotatable Bonds"].append(rot_bonds)
        property_valid_lst["Aromatic Rings"].append(aromatic_rings)
        property_valid_lst["Heavy Atoms"].append(heavy_atoms)
        property_valid_lst["SA Score"].append(sa_score)
        property_valid_lst["QED Score"].append(qed_score)

with open("property_valid_lst_generate.pkl", "wb") as file:
    pickle.dump(property_valid_lst, file)
        