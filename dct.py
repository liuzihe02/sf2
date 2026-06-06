import matplotlib.pyplot as plt
from cued_sf2_lab.dct import dct_ii
import numpy as np
from cued_sf2_lab.familiarisation import load_mat_img
from cued_sf2_lab.dct import colxfm
from cued_sf2_lab.familiarisation import plot_image
from cued_sf2_lab.dct import regroup
from cued_sf2_lab.laplacian_pyramid import bpp
from cued_sf2_lab.laplacian_pyramid import quantise

def bitsdct(step, rise1):
    C8 = dct_ii(8)
    N = 8

    X_pre_zero_mean, cmaps_dict = load_mat_img(img='lighthouse.mat', img_info='X', cmap_info={'map', 'map2'})
    X = X_pre_zero_mean - 128.0

    Y = colxfm(colxfm(X, C8).T, C8).T
    Z = colxfm(colxfm(Y.T, C8.T).T, C8.T)

    Yq = quantise(Y, step, rise1)
    Yr = regroup(Yq, N)

    def dctbpp(Yr, N):
        bits = 0
        step_size = Yr.shape[0] // N
        # split Yr in NxN blocks and calculate the bpp for each block
        for i in range(0, Yr.shape[0], step_size):
            for j in range(0, Yr.shape[1], step_size):
                block = Yr[i:i+step_size, j:j+step_size]
                bits += bpp(block) * step_size**2
        return bits

    bits = dctbpp(Yr, N)

    return bits

    Zq = colxfm(colxfm(Yq.T, C8.T).T, C8.T)

    RMS = np.sqrt(np.mean((Zq - X)**2))
    print("QuantisedRMS: ", RMS)