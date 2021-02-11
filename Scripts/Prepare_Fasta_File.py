from Bio import SeqIO

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import os

import pandas as pd
version = 7
folder_path = r'C:\Users\sajid\Downloads\CASP\CASP'+str(version)
csv_path = folder_path + '\casp'+str(version)+'_dataset.csv'

df = pd.read_csv(csv_path)


def remove_new_lines(s):
    new_s =''
    for letters in s:
        if (letters == '\n'):
            continue
        new_s += letters
    return new_s
# strings = 'A\nB\nC'
# print(remove_new_lines(strings))
# exit()

print(df.columns)
for i in range(0,len(df)):
    name = df.iloc[i]['Name']
    Sequence = df.iloc[i]['Sequence']
    # print(Sequence)
    Sequence = Sequence.replace('\n','')
    # Sequence = Sequence.rstrip()
    # Sequence = remove_new_lines(Sequence)
    record = SeqRecord(
        Seq(Sequence),
        id=name,
        name="",
        description="",
    )
    # print(record.seq)
    # print('\n')
    if not os.path.exists(folder_path + '\\fasta\\'):
        os.mkdir(folder_path + '\\fasta\\')
    SeqIO.write(record, folder_path + '\\fasta\\'+str(name) + '.fasta', "fasta-2line")


record = SeqIO.read(r"C:\Users\sajid\Downloads\CASP\CASP14\fasta\T1024.fasta", "fasta")
print(record.seq)




