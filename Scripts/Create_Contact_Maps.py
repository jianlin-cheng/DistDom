import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from numpy import inf
import cv2
# def equ(r, c = 7):
#     return max(1-r/c,0)
#
# similarity_scores = []
# for i in range(0,11):
#     print(i, equ(i))
#     similarity_scores.append(equ(i))
# plt.plot(similarity_scores)
# plt.xlabel('r')
# plt.ylabel('Similarity Score')
# plt.show()
path = r'C:\Users\sajid\Downloads\CASP\distance_maps\CASP14'
save_path = r'C:\Users\sajid\Downloads\CASP\distance_maps_visualizations'
# mat = np.load(path)
# mat = np.squeeze(mat)
# print(len(mat))
# mat = np.reciprocal(mat)
# norm = np.linalg.norm(mat)
# mat = mat/norm
# tmp = np.argmax(mat,axis=2)
# tmp = ((tmp * 1)/24).astype(np.float64)
# print(mat[0][0])
def get_contact_map(mat):
    contact_map = np.empty([len(mat), len(mat)])
    for i in range(0, len(mat)):
        for j in range(0, len(mat)):
            sum = 0
            for k in range(0, 3):
                sum += mat[i][j][k]
            print(sum)
            contact_map[i][j] = '%.3f' % sum
    return contact_map

for filename in os.listdir(path):
    if filename.endswith(".npy"):
        file_path = path + '\\' + filename
        mat = np.squeeze(np.load(file_path))
        contact_map = get_contact_map(mat)
        image = Image.fromarray(np.uint8(contact_map * 255), 'L')

         # image = Image.fromarray(contact_map * 255 , 'L')

        image.save(r'C:\Users\sajid\Downloads\CASP\distance_maps_visualizations\{0}'.format(str(filename)+'.png'), format='PNG')

    else:
        continue

# contact_map = get_contact_map(mat)



# image = Image.fromarray(np.uint8(contact_map * 255) , 'L')
#
# # image = Image.fromarray(contact_map * 255 , 'L')
#
#
#
#
# image.save(r'C:\Users\sajid\Downloads\CASP\distance_maps\demo\demo.png',format='PNG')


# real_dist_path = r'C:\Users\sajid\Downloads\CASP\distance_maps\T1024\full_length\ensemble\pred_map_ensem\real_dist\T1024.txt'
#
# real_mat = np.loadtxt(real_dist_path)
# real_mat = np.reciprocal(real_mat)
# real_mat[real_mat == inf] = 1
#
# image = Image.fromarray(np.uint8(real_mat * 255) , 'L')
# image.save(r'C:\Users\sajid\Downloads\CASP\distance_maps\demo\demo_real.png',format='PNG')




