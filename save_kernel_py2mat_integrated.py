'''
Script for saving pytorch model's kernels to mat extension. 
This script converts kernels from polar to cartesian. 
'''


import torch
import torch.nn as nn
import scipy.io
from convert_polar2cart import *

'''
Control variables
 - isConvert2Cart: if set True, convert kernel from polar-CNN to cartesian space
 - isEmbedKernel: if set True, embed polar kernel to polar image space
 if isConvert2Cart is False, isEmbedKernel must be ignored.
'''
isConvert2Cart = False#True
isEmbedKernel = True

path_state_dict = './Attn_e41b0.pt'
# Resolution of Polar coordinates when training polar-cnn
resPolar = 224
resCart = 224
divide_logr = 5



trained_dict = torch.load(path_state_dict)

for k, v in list(trained_dict.items()):
    print(k.split('.'))
    if k.split('.')[2] == 'conv1':
        kernel = v

print('Loaded Kernel Size:  ', kernel.size())









s_dict = {}
s_dict['conv1_k'] = kernel.cpu().numpy()
if isConvert2Cart:
    if isEmbedKernel:
        grid = polar2cartesian_grid((resCart,resCart), batch_s=kernel.size(0))
        for i in range(divide_logr):
            k_min = torch.min(kernel)
            canvas = torch.ones([kernel.size(0), kernel.size(1), resPolar, resPolar], dtype=torch.float).cuda()
            canvas = canvas + k_min
            print(int(resPolar/divide_logr)*i, int(resPolar/divide_logr)*i+kernel.size(3))
            canvas[:, :, int(resPolar/2):int(resPolar/2)+kernel.size(2), int(resPolar/divide_logr)*i:int(resPolar/divide_logr)*i+kernel.size(3)] = kernel
            print(torch.max(canvas), torch.min(canvas))
            kernel_p = nn.functional.grid_sample(canvas, grid)
            print(torch.max(kernel_p), torch.min(kernel_p))
            k_name = 'conv1_k_p' + str(i)
            s_dict[k_name] = kernel_p.cpu().numpy()
        canvas = torch.ones([kernel.size(0), kernel.size(1), resPolar, resPolar], dtype=torch.float).cuda()
        canvas = canvas + k_min
        print(int(resPolar/divide_logr)*i, int(resPolar/divide_logr)*i+kernel.size(3))
        canvas[:, :, int(resPolar/2):int(resPolar/2)+kernel.size(2), resPolar-7-15:-15] = kernel
        print(torch.max(canvas), torch.min(canvas))
        kernel_p = nn.functional.grid_sample(canvas, grid)
        print(torch.max(kernel_p), torch.min(kernel_p))
        kernel_p.cpu().numpy()
        s_dict['conv1_k_p_last'] = kernel_p.cpu().numpy()
    else:
        grid = polar2cartesian_grid((10,10), batch_s=kernel.size(0))
        kernel_p = nn.functional.grid_sample(kernel, grid)
        s_dict['conv1_k_p'] = kernel_p.cpu().numpy()
scipy.io.savemat('kernel.mat', s_dict)
