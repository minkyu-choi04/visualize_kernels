import torch
import numpy as np


def polar2cartesian_grid(output_size, ylim=(-1., 1.), xlim=(-1., 1.), out=None, device=None, batch_s=1):
    ny, nx = output_size
    y_range = torch.linspace(ylim[0], ylim[1], ny).cuda()
    x_range = torch.linspace(xlim[0], xlim[1], nx).cuda()
    ys, xs = torch.meshgrid([y_range, x_range])
    logrs = torch.log(torch.pow(ys,2) + torch.pow(xs,2))/2.
    thetas = torch.atan(ys/xs)
    ret = torch.stack([thetas, logrs], 2, out=out)
    return ret.unsqueeae(0).repeat(batch_s, 1, 1, 1)

'''
    nv, nu = output_size
    urange = torch.linspace(ulim[0], ulim[1], nu).cuda()
    vrange = torch.linspace(vlim[0], vlim[1], nv).cuda()
    vs, us = torch.meshgrid([vrange, urange])
    rs = torch.exp(us)
    xs = rs * torch.cos(vs)
    ys = rs * torch.sin(vs)
    ret = torch.stack([xs, ys], 2, out=out)
    return ret.unsqueeze(0).repeat(batch_s, 1, 1, 1)
'''
