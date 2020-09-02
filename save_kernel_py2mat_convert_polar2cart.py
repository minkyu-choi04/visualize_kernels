'''
Script for saving pytorch model's kernels to mat extension. 
This script converts kernels from polar to cartesian. 
'''


import torch
import torch.nn as nn
import scipy.io
from convert_polar2cart import *

trained_dict = torch.load('./Attn_e41b0.pt')

for k, v in list(trained_dict.items()):
    print(k.split('.'))
    if k.split('.')[2] == 'conv1':
        kernel = v

print(kernel.size())


grid = polar2cartesian_grid((10,10), batch_s=kernel.size(0))
kernel_p = nn.functional.grid_sample(kernel, grid)


s_dict = {}
s_dict['conv1_k'] = kernel.cpu().numpy()
s_dict['conv1_k_p'] = kernel_p.cpu().numpy()
scipy.io.savemat('kernel.mat', s_dict)
