import torch
import torch.nn as nn
import torch.nn.functional

def polar2cartesian_grid(output_size, ylim=(-1., 1.), xlim=(-1., 1.), out=None, device=None, batch_s=1):
    ny, nx = output_size
    y_range = torch.linspace(ylim[0], ylim[1], ny)
    x_range = torch.linspace(xlim[0], xlim[1], nx)
    ys, xs = torch.meshgrid([y_range, x_range]) # (200x200)
    logrs = torch.log(torch.pow(ys,2) + torch.pow(xs,2))/2. 
    print(ys.size())
    thetas = torch.atan2(ys,xs) 

    ## normalize ##
    thetas = thetas / torch.max(thetas)# * 0.9
    logrs = (logrs - (torch.max(logrs) + torch.min(logrs))/2.) / (torch.max(logrs) - torch.min(logrs)) *2

    print(thetas, thetas.size())
    #ret = torch.stack([thetas, logrs], 2, out=out)
    ret = torch.stack([logrs, thetas], 2, out=out)
    ret = ret.cuda()
    return ret.unsqueeze(0).repeat(batch_s, 1, 1, 1)




