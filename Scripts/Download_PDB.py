import os
import urllib.request
import pandas as pd
import sys
CASP_NAME = 'CASP9'
target_name_path = r'C:\Users\sajid\Downloads\CASP\distance_maps\{0}'.format(CASP_NAME)
pdb_name_path =  r'{0}/Domains Deffinition Summary - {0}.csv'.format(CASP_NAME,CASP_NAME)
pdb_save_path = r'C:\Users\sajid\Downloads\CASP\PDB Files\{0}'.format(CASP_NAME)

if not os.path.exists(pdb_save_path):
    os.mkdir(pdb_save_path)

def download_pdb(pdbcode, datadir,target_name, downloadurl="https://files.rcsb.org/download/"):
    """
    Downloads a PDB file from the Internet and saves it in a data directory.
    :param pdbcode: The standard PDB ID e.g. '3ICB' or '3icb'
    :param datadir: The directory where the downloaded file will be saved
    :param downloadurl: The base PDB download URL, cf.
        `https://www.rcsb.org/pages/download/http#structures` for details
    :return: the full path to the downloaded PDB file or None if something went wrong
    """
    pdbfn = pdbcode + ".pdb"
    url = downloadurl + pdbfn
    # outfnm = os.path.join(datadir, pdbfn)
    outfnm = os.path.join(datadir,target_name)
    try:
        urllib.request.urlretrieve(url, outfnm)
        return outfnm
    except Exception as err:
        print(str(err), file=sys.stderr)
        return None

df = pd.read_csv(pdb_name_path)
for filename in os.listdir(target_name_path):
    if filename.endswith('.npy'):
        target_name = filename.split('.')[0]
        # print(target_name)
        pdb_name = df.loc[df['Target'] == target_name, 'PDB'].iloc[0]
        if pdb_name != '-':
            print(target_name)
            print(pdb_name)
            print(download_pdb(pdb_name,pdb_save_path,target_name))
    else:
        continue