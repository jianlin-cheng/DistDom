
from pathlib import Path
import pandas as pd
sequence_path = r'C:\Users\sajid\Downloads\CASP\CASP9\casp9.seq.txt'
domain_definition_csv_path = r'C:\Users\sajid\Downloads\CASP\CASP9\Domains Deffinition Summary - CASP9.csv'
save_path = r'C:\Users\sajid\Downloads\CASP\CASP9\casp9_dataset.csv'
file = open(sequence_path, mode = 'r', encoding = 'utf-8-sig')
lines = file.readlines()
text = file.read()
df = pd.read_csv(domain_definition_csv_path)
column_list = list(df.columns)
print(column_list)
df.drop(df.loc[df[column_list[6]]== 'MultiDom'].index, inplace=True)
df.drop(df.loc[df[column_list[5]]== 'Canceled'].index,inplace= True)

file.close()

text = Path(sequence_path).read_text()


first_value = df[df.columns[4]].iloc[0]
# print(first_value.split(':')[0].split('-'))

parse = text.split('>')
count = 0
protein_name_list = []
sequence_name_list = []
domains_list = []
labels_list = []
boundary_region_list = []
number_of_residues_list = []
for idx, contents in enumerate(parse):
    if idx == 0:
        continue
    split_contents = contents.split(';\n')
    protein_name = split_contents[0].split(',')[0].split(' ')[0]
    number_of_residues = split_contents[0].split(',')
    sequence = split_contents[1].strip('\n')
    print(protein_name)
    print(len(sequence))
    domains = df.loc[df[column_list[1]] == protein_name, column_list[4]]
    boundary_region_string = ''
    dom_count = len(domains)
    if len(domains!=0):
        previous_dom_end = -1
        domain_definition_list = []
        for i in range(0,len(domains)):
            if(domains.iloc[i].split(': ')[0].split('-')[1]=='D0'):
                continue
            dom_name = domains.iloc[i].split(': ')[0].split('-')[1]
            dom_range = domains.iloc[i].split(': ')[1].split(',')

            for dom in dom_range:

                dom_start = int(dom.split('-')[0])
                dom_end = int(dom.split('-')[1])
                domain_definition_list.append(int(dom_start))
                domain_definition_list.append(int(dom_end))

                print(dom_name, dom_start, dom_end)
        domain_definition_list_sorted = sorted(domain_definition_list)
        if (domain_definition_list_sorted[0]-0 < 40):
            domain_definition_list_sorted[0] = 1
        else:
            temp = domain_definition_list_sorted[0]
            domain_definition_list_sorted[0] += 5
            domain_definition_list_sorted.append(1)
            domain_definition_list_sorted.append(temp - 6)
            domain_definition_list_sorted = sorted(domain_definition_list_sorted)
            dom_count += 1
        if (len(sequence)-domain_definition_list_sorted[-1] < 40):
            domain_definition_list_sorted[-1] = len(sequence)
        else:
            temp = domain_definition_list_sorted[-1]
            domain_definition_list_sorted[-1] -= 5
            domain_definition_list_sorted.append(temp + 6)
            domain_definition_list_sorted.append(len(sequence))
            domain_definition_list_sorted = sorted(domain_definition_list_sorted)
            dom_count += 1
        print(domain_definition_list_sorted)

        for j in range (0, len(domain_definition_list_sorted)):
            if (j==0 and domain_definition_list_sorted[j]!=1):
                print(str(1) + '-' + str(domain_definition_list_sorted[j]))
                boundary_region_string += str(1) + 'to' + str(domain_definition_list_sorted[j] - 1) + ' '
            if (j == len(domain_definition_list_sorted)-1 and domain_definition_list_sorted[j] != len(sequence)):
                print(str(domain_definition_list_sorted[j] + 1) + '-' + str(len(sequence)))
                boundary_region_string += str(domain_definition_list_sorted[j] + 1) + 'to' + str(len(sequence)) + ' '

            if(j%2!=0 and j != len(domain_definition_list_sorted)-1):
                if(domain_definition_list_sorted[j] == domain_definition_list_sorted[j + 1]-1):
                    domain_definition_list_sorted[j] -= 5
                    domain_definition_list_sorted[j + 1] += 5
                print(str(domain_definition_list_sorted[j] + 1) + '-' + str(domain_definition_list_sorted[j + 1] - 1))
                boundary_region_string += str(domain_definition_list_sorted[j] + 1) + 'to' + str(domain_definition_list_sorted[j + 1] - 1) + ' '

        print(domain_definition_list_sorted)
        if(dom_count < 2):
            boundary_region_string = 'No boundaries'
        print(boundary_region_string)

        # print(sequence)
        label = [1]*len(sequence)
        for j in range(0, len(domain_definition_list_sorted)):
            if (j%2==0):
                for k in range(domain_definition_list_sorted[j], domain_definition_list_sorted[j + 1] + 1):
                    label[k-1] -= 1
        if (dom_count < 2):
            label = [0] * len(sequence)
        print(label)
        protein_name_list.append(protein_name)
        sequence_name_list.append(sequence)
        labels_list.append(label)
        domains_list.append(domains)
        boundary_region_list.append(boundary_region_string)
        number_of_residues_list.append(len(sequence))



    count += 1


print(count)
print(protein_name_list[0])
print(sequence_name_list[0])
print(domains_list[0])
print(labels_list[0])


final_df = pd.DataFrame(list(zip(protein_name_list, domains_list, boundary_region_list, number_of_residues_list, sequence_name_list, labels_list)),
               columns =['Name', 'Domains', 'Boundary Range', 'Number of Residues','Sequence', 'Label'])
final_df.to_csv(save_path)


