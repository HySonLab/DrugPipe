import pandas as pd
chembl_bank     = pd.read_csv('../drugbank_downloader/chembl_34_chemreps.txt', delimiter= "\t")
print(chembl_bank['smiles'])