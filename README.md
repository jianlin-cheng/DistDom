# DistDom
Deep learning prediction of protein domains from distance maps
# Installation
1. Install anaconda from https://www.anaconda.com/products/individual
2. Open Terminal
3. Create conda environment using the following script: conda create --name DistDom
4. Activate conda environment: conda activate DistDom
5. Now install the following python packages: PyTorch, Scikit-learn, glob, numpy
# Prediction
make sure the environment is activated. Go to the directory, use the following command to run predictions:
python -W ignore make_predictions.py --device "cpu or gpu" --distance_map /path/to/distancemaps --seq /path/to/1D_sequence --label /path/to/labels
1. Set device to 'cpu' or 'cuda' according to your need
2. Set --distance_map to the path of your stored distance maps
3. Set --seq to the path of your stored 1d features
4. Set --label to the path of your stored ground truths

# Prediction Example
python -W ignore make_predictions.py --device cpu --distance_map distance_map --seq 1D_sequence --label labels

Predictions will be saved in the dir Saved_Predictions
