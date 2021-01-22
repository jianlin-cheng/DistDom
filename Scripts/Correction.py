import pandas as pd

path = r'C:\Users\sajid\Downloads\CASP\CASP12\casp12_dataset.csv'
df = pd.read_csv(path)
print(df.columns)
Names = ['T0899', 'T0914', 'T0923','T0941']
Wrong_Boundary = [['123to129','169to178'],['79to92'],['92to94','137to157','264to319'],['243to246']]
for idx,name in enumerate(Names):
    print(name)
    # print(df.loc[df['Name']==name,'Domains'])
    list_of_boundary = df.loc[df['Name'] == name, 'Boundary Range'].iloc[0].split(' ')
    number_of_residues = df.loc[df['Name'] == name, 'Number of Residues'].iloc[0]
    label = df.loc[df['Name'] == name, 'Label'].iloc[0]
    print(number_of_residues)
    print(label)
    corrected_label = [0]* int(number_of_residues)
    list_of_boundary.pop(-1)
    for wrongs in Wrong_Boundary[idx]:
        list_of_boundary.remove(wrongs)
    print(list_of_boundary)
    boundary_string = ''
    for boundaries in list_of_boundary:
        start = int(boundaries.split('to')[0])
        end = int(boundaries.split('to')[1])
        for indexes in range(start-1,end):
            corrected_label[indexes] += 1
        boundary_string += boundaries + ' '
    print(corrected_label)
    df.loc[df['Name'] == name, 'Boundary Range'] = boundary_string
    df.loc[df['Name'] == name, 'Label'] = str(corrected_label)
print(df.loc[df['Name'] == Names[0], 'Label'])
df.to_csv(path)








