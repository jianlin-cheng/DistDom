#! /usr/bin/env python
import os
from os import listdir
from os.path import isfile, join
casp = 'CASP7'
filepath = r'C:\Users\sajid\Downloads\CASP\{0}\fasta'.format(casp)
onlyfiles = [f for f in listdir(filepath) if isfile(join(filepath, f))]
file_without_ext = [f.split('.')[0] for f in onlyfiles]
shell_script_path = r'C:\Users\sajid\Downloads\CASP\{0}\Shell_Scripts'.format(casp)
if not os.path.exists(r'C:\Users\sajid\Downloads\CASP\{0}\Shell_Scripts'.format(casp)):
    os.mkdir(r'C:\Users\sajid\Downloads\CASP\{0}\Shell_Scripts'.format(casp))




for file in onlyfiles:
    with open (shell_script_path + '\\run_deepdist_'+str(file.split('.')[0])+'.sh', 'w') as rsh:
        rsh.write("""#!/bin/bash -l
#!/bin/bash -l
#SBATCH -J  PRED
#SBATCH -o PRED-%j.out
#SBATCH -p Lewis,hpc4,hpc5
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 2-00:00
#SBATCH --mem 200G

module load cuda/cuda-9.0.176
module load cudnn/cudnn-7.1.4-cuda-9.0.176
export GPUARRAY_FORCE_CUDA_DRIVER_LOAD=""
export HDF5_USE_FILE_LOCKING=FALSE
temp_dir=$(pwd)
##GLOBAL_FLAG
global_dir=/storage/htc/bdm/zhiye/DeepDist
## ENV_FLAG
source $global_dir/env/deepdist_virenv/bin/activate

python $global_dir/lib/run_deepdist.py -f /storage/htc/bdm/sajid/CASP7/fasta/{0} -o /storage/htc/bdm/sajid/distance_maps/CASP7/{1} -df 1 -m 'mul_class_C_11k'""".format(file,file.split('.')[0]))