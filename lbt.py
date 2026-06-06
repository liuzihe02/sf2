import warnings
import inspect
import matplotlib.pyplot as plt
import IPython.display
import numpy as np
from cued_sf2_lab.familiarisation import load_mat_img, plot_image
from cued_sf2_lab.dct import colxfm
from scipy.optimize import minimize_scalar
from cued_sf2_lab.laplacian_pyramid import bpp
from cued_sf2_lab.laplacian_pyramid import quantise
from cued_sf2_lab.dct import regroup

X, cmaps_dict = load_mat_img(img='lighthouse.mat', img_info='X', cmap_info={'map', 'map2'})
X = X - 128.0

from cued_sf2_lab.lbt import pot_ii

# your code here
from cued_sf2_lab.dct import dct_ii
from cued_sf2_lab.laplacian_pyramid import quantise

C8 = dct_ii(8)

def forward_lbt(X,s=1, N=8):
    Pf, Pr = pot_ii(N,s)
    t = np.s_[N//2:-N//2]  # N is the DCT size, I is the image size
    Xp = X.copy()  # copy the non-transformed edges directly from X
    Xp[t,:] = colxfm(Xp[t,:], Pf)
    Xp[:,t] = colxfm(Xp[:,t].T, Pf).T
    return Xp

def inverse_lbt(Zp,s=1, N=8):
    Pf, Pr = pot_ii(N,s)
    t = np.s_[N//2:-N//2]  # N is the DCT size, I is the image size
    Z = Zp.copy()  #copy the non-transformed edges directly from Z
    Z[:,t] = colxfm(Z[:,t].T, Pr.T).T
    Z[t,:] = colxfm(Z[t,:], Pr.T)
    return Z

def dctbpp(Yr, N):
    bits = 0
    step_size = Yr.shape[0] // N
    # split Yr in NxN blocks and calculate the bpp for each block
    for i in range(0, Yr.shape[0], step_size):
        for j in range(0, Yr.shape[1], step_size):
            block = Yr[i:i+step_size, j:j+step_size]
            bits += bpp(block) * step_size**2
    return bits

def bitslbt(step, rise1):
    s = np.sqrt(2)
    Xps = forward_lbt(X, s, N=len(C8))
    Ys = colxfm(colxfm(Xps, C8).T, C8).T
    Yq = quantise(Ys, step,rise1)
    Yr = regroup(Yq, len(C8))
    bitsdct = dctbpp(Yr, 16)
    return bitsdct