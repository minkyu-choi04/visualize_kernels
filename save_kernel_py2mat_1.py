import torch
import scipy.io

trained_dict = torch.load('./Attn_e132b0.pt')

for k, v in list(trained_dict['state_dict'].items()):
    print(k.split('.'))
    if k.split('.')[2] == 'conv1':
        kernel = v

print(kernel.size())

s_dict = {}
s_dict['conv1_k'] = kernel.cpu().numpy()
scipy.io.savemat('kernel.mat', s_dict)
