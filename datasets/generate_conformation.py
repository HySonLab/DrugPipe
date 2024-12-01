from molvs import Standardizer
from rdkit import Chem
from rdkit.Chem import SaltRemover, AllChem, Descriptors
import pandas as pd
import os, time

### Database ID, SMILES
cnt = 0
cmpd_collect = {}
drugbank_id_list =[]
chembl_id_list =[]


def conformer_generation():
    cnt = 0
    df = pd.read_csv('drugbank.csv')

    for i in range(len(df)):
        row = df.loc[i]
        smi = row['smiles']
        key_id = row['Drug id']
        try:
            m = Chem.MolFromSmiles(smi)
            m2 = Chem.AddHs(m)
            mw = Descriptors.ExactMolWt(m)
            print("cid: ", key_id)
            print("MW: ", mw)
            if mw > 1000:
                continue

            # run ETKDG 300 times
            cids = AllChem.EmbedMultipleConfs(m2, numConfs=300, numThreads=1)

            nMol = len(cids)
            print("num of conformers: ", nMol)
            try:
                os.mkdir("./drugbank_conformation")
            except:
                pass
            w = Chem.SDWriter(f'./drugbank_conformation/{key_id}.sdf')
            for prbNum in range(0, nMol):
                prbMol = cids[prbNum]
                w.write(m2, prbMol)
            w.close()
            print('Success: cnt %s %s' %(cnt, key_id))
            cnt += 1
        except:
            print('error')
            pass
    return

conformer_generation()