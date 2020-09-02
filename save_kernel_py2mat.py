'''
This script is for saving pytorch model's first layer kernels to mat extension. 
This kernel is not going through the cartesian-polar transformation. 
To convert original kernel to cartesian, see 
    save_kernel_py2mat_convert_polar2cart.py 
'''


import torch
import scipy.io

trained_dict = torch.load('./Attn_e41b0.pt')

for k, v in list(trained_dict.items()):
    print(k.split('.'))
    if k.split('.')[2] == 'conv1':
        kernel = v

print(kernel.size())

s_dict = {}
s_dict['conv1_k'] = kernel.cpu().numpy()
scipy.io.savemat('kernel.mat', s_dict)
