import os
import numpy as np

import hickle as hkl


import IPython

# run in wsk PC, not in docker

# Data files
data_dir = '../../kitti_data/'
train_file = os.path.join(data_dir, 'X_train.hkl')
train_sources = os.path.join(data_dir, 'sources_train.hkl')
val_file = os.path.join(data_dir, 'X_val.hkl')
val_sources = os.path.join(data_dir, 'sources_val.hkl')
test_file = os.path.join(data_dir, 'X_test.hkl')
test_sources = os.path.join(data_dir, 'sources_test.hkl')

X = hkl.load(train_file)
S = hkl.load(train_sources)
save_dir = '/home/skwang/data/dataset_kitti'
save_train_dir = os.path.join(save_dir, 'train_bin.npz')
np.savez(save_train_dir, X, S)
print("save bin file into " + save_train_dir)
print("finish the train data process.")

X = hkl.load(val_file)
S = hkl.load(val_sources)
save_dir = '/home/skwang/data/dataset_kitti'
save_val_dir = os.path.join(save_dir, 'val_bin.npz')
np.savez(save_val_dir, X, S)
print("save bin file into " + save_val_dir)
print("finish the val data process.")

X = hkl.load(test_file)
S = hkl.load(test_sources)
save_dir = '/home/skwang/data/dataset_kitti'
save_test_dir = os.path.join(save_dir, 'test_bin.npz')
np.savez(save_test_dir, X, S)
print("save bin file into " + save_test_dir)
print("finish the test data process.")
