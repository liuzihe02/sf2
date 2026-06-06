import warnings
import inspect
import matplotlib.pyplot as plt
import IPython.display
from cued_sf2_lab.familiarisation import load_mat_img, plot_image
import numpy as np
from typing import Tuple
from cued_sf2_lab.laplacian_pyramid import quantise, bpp
from cued_sf2_lab.dwt import idwt
from cued_sf2_lab.dwt import dwt

def bitsdwt(step, rise1):
    X, _ = load_mat_img(img='lighthouse.mat', img_info='X', cmap_info={'map', 'map2'})
    X = X - 128.0
    h1 = np.array([-1, 2, 6, 2, -1])/8
    h2 = np.array([-1, 2, -1])/4

    from cued_sf2_lab.laplacian_pyramid import rowdec
    U = rowdec(X, h1)

    from cued_sf2_lab.laplacian_pyramid import rowdec2
    V = rowdec2(X, h2)

    UU = rowdec(U.T, h1).T
    UV = rowdec2(U.T, h2).T
    VU = rowdec(V.T, h1).T
    VV = rowdec2(V.T, h2).T

    from cued_sf2_lab.laplacian_pyramid import rowint, rowint2

    g1 = np.array([1, 2, 1])/2
    g2 = np.array([-1, -2, 6, -2, -1])/4
    Ur = rowint(UU.T, g1).T + rowint2(UV.T, g2).T
    Vr = rowint(VU.T, g1).T + rowint2(VV.T, g2).T
    Xr = rowint(Ur,g1) + rowint2(Vr,g2)
    # your code here
    block = np.block([[UU, UV], [VU, VV]]) * 4


    def nlevdwt(X, n):
        m = 256
        Y = dwt(X)
        for i in range(n):
            m = m // 2
            Y[:m, :m] = dwt(Y[:m, :m])
        return Y

    def nlevidwt(Y, n):
        m = int(256 // (2**n))
        for i in range(n+1):
            Xr = idwt(Y[:m, :m])
            Y[:m, :m] = Xr
            m = m * 2
        return Xr

    def quantdwt(Y: np.ndarray, dwtstep: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parameters:
            Y: the output of `dwt(X, n)`
            dwtstep: an array of shape `(3, n+1)`
        Returns:
            Yq: the quantized version of `Y`
            dwtenc: an array of shape `(3, n+1)` containing the entropies
        """


        n_levels = len(dwtstep[0])
        m = 256
        dwtent = np.zeros((3, n_levels))
        for i in range(n_levels - 1):
            m = m // 2
            # Quantise each sub-image of Y now
            Y[:m, :m] = quantise(Y[:m, :m], dwtstep[0][i],rise1) # low pass image
            # top right quadrant
            Y[:m, m:2*m] = quantise(Y[:m, m:2*m], dwtstep[0][i],rise1)
            dwtent[0][i] = bpp(Y[:m, m:2*m]) * m**2
            # bottom left quadrant
            Y[m:2*m, :m] = quantise(Y[m:2*m, :m], dwtstep[1][i],rise1)
            dwtent[1][i] = bpp(Y[m:2*m, :m]) * m**2
            # bottom right quadrant
            Y[m:2*m, m:2*m] = quantise(Y[m:2*m, m:2*m], dwtstep[2][i],rise1)
            dwtent[2][i] = bpp(Y[m:2*m, m:2*m]) * m**2
            
            """
            fig, axs = plt.subplots(1, 4, figsize=(12, 4))
            plot_image(Y[:m, :m], ax=axs[0])
            axs[0].set(title="Low pass")
            plot_image(Y[:m, m:2*m], ax=axs[1])
            axs[1].set(title="Top right")
            plot_image(Y[m:2*m, :m], ax=axs[2])
            axs[2].set(title="Bottom left")
            plot_image(Y[m:2*m, m:2*m], ax=axs[3])
            axs[3].set(title="Bottom right")
            fig.tight_layout()
            plt.show()
            """
            
            Y[:m, :m] = dwt(Y[:m, :m])

        dwtent[0][-1] = bpp(Y[:m, :m]) * m**2

        #print(dwtent)
        return Y, dwtent

    # 5x3 array of value step for all values 
    dwtstep = np.full((3,5),step)
    Y, dwtent = quantdwt(dwt(X), dwtstep)
    return np.sum(dwtent)