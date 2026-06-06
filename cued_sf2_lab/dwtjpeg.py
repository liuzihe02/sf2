"""
DWT-based JPEG-style encoder/decoder.

Mirrors cued_sf2_lab/jpeg.py but replaces the block DCT with a multi-level
DWT.  The nlevdwt / nlevidwt helpers follow the EXACT convention used in
sf2/9-dwt.ipynb:

    nlevdwt(X, n)   applies n+1 total DWT levels  (1 initial + n in loop)
    nlevidwt(Y, n)  applies n+1 inverse DWT levels

so dwtgroup is called with n+1, giving 2^(n+1) x 2^(n+1) coding blocks.

Quantisation uses a dwtstep array of shape (3, n+1) — same layout as
quantdwt in the notebook — so each sub-band level can have its own step
size.  The last column dwtstep[:, n] is used for the deepest LL sub-band.
"""
from typing import Optional, Tuple

import numpy as np

from .laplacian_pyramid import quant1, quant2
from .dwt import dwt, idwt
from .jpeg import (
    HuffmanTable,
    diagscan,
    dwtgroup,
    huffdes,
    huffdflt,
    huffenc,
    huffgen,
    runampl,
)

__all__ = [
    "nlevdwt", "nlevidwt",
    "quantise_subbands", "dequantise_subbands",
    "dwtjpegenc", "dwtjpegdec",
    "dwt_equal_mse_ratios",
]


# ---------------------------------------------------------------------------
# Multi-level DWT helpers — copied from 9-dwt.ipynb so behaviour is
# identical to the notebook.
# ---------------------------------------------------------------------------

def nlevdwt(X: np.ndarray, n: int) -> np.ndarray:
    """
    Apply n+1 DWT levels to X (1 initial level plus n more on the LL band).

    Matches nlevdwt from 9-dwt.ipynb exactly.  After this call the deepest
    LL sub-band sits at Y[:H//2^(n+1), :W//2^(n+1)].
    """
    m = X.shape[0]
    Y = dwt(X)
    for _ in range(n):
        m //= 2
        Y[:m, :m] = dwt(Y[:m, :m])
    return Y


def nlevidwt(Y: np.ndarray, n: int) -> np.ndarray:
    """
    Apply n+1 inverse DWT levels — exact inverse of nlevdwt(X, n).

    Matches nlevidwt from 9-dwt.ipynb.  Works on a copy so the input is
    not modified.
    """
    Z = Y.copy()
    m = Z.shape[0] // (2 ** n)
    for _ in range(n + 1):
        Xr = idwt(Z[:m, :m])
        Z[:m, :m] = Xr
        m *= 2
    return Xr


# ---------------------------------------------------------------------------
# Per-sub-band quantisation helpers
# ---------------------------------------------------------------------------

def quantise_subbands(Y: np.ndarray, dwtstep: np.ndarray, n: int) -> np.ndarray:
    """
    Quantise each DWT sub-band to integer step indices.

    Y must be the output of nlevdwt(X, n), so it has n+1 total DWT levels.
    dwtstep has shape (3, n+1):
        dwtstep[0, i]  step for LH sub-band at level i+1
        dwtstep[1, i]  step for HL sub-band at level i+1
        dwtstep[2, i]  step for HH sub-band at level i+1
        dwtstep[0, n]  step for the deepest LL sub-band

    Returns an integer array of the same shape as Y.
    """
    H = Y.shape[0]
    Yq = Y.copy()
    m = H
    for i in range(n + 1):
        m //= 2
        step_lh = dwtstep[0, i]
        step_hl = dwtstep[1, i]
        step_hh = dwtstep[2, i]
        # LH: top strip, right half
        Yq[:m, m:2*m] = quant1(Y[:m, m:2*m], step_lh)
        # HL: bottom strip, left half
        Yq[m:2*m, :m] = quant1(Y[m:2*m, :m], step_hl)
        # HH: bottom-right block
        Yq[m:2*m, m:2*m] = quant1(Y[m:2*m, m:2*m], step_hh)
    # Deepest LL sub-band (sits at Y[:m, :m] after the last iteration)
    Yq[:m, :m] = quant1(Y[:m, :m], dwtstep[0, n])
    return Yq.astype('int')


def dequantise_subbands(Yq: np.ndarray, dwtstep: np.ndarray, n: int) -> np.ndarray:
    """
    Reconstruct float sub-band values from integer step indices.

    Exact inverse of _quantise_subbands.
    """
    H = Yq.shape[0]
    Y = Yq.astype(float)
    m = H
    for i in range(n + 1):
        m //= 2
        Y[:m, m:2*m] = quant2(Yq[:m, m:2*m], dwtstep[0, i])
        Y[m:2*m, :m] = quant2(Yq[m:2*m, :m], dwtstep[1, i])
        Y[m:2*m, m:2*m] = quant2(Yq[m:2*m, m:2*m], dwtstep[2, i])
    Y[:m, :m] = quant2(Yq[:m, :m], dwtstep[0, n])
    return Y


# ---------------------------------------------------------------------------
# Equal-MSE step-size ratios
# ---------------------------------------------------------------------------

def dwt_equal_mse_ratios(n: int, H: int = 256) -> np.ndarray:
    """
    Return normalised step-size ratios for equal-MSE quantisation of an
    nlevdwt(X, n) transform on an H×H image (n+1 total DWT levels).

    Method: place a unit impulse at the centre of each sub-band, apply
    nlevidwt, and measure the RMS energy in the reconstructed image.
    Sub-bands with larger energy gain should receive proportionally larger
    step sizes so that each sub-band contributes equally to the total MSE.

    Returns
    -------
    ratios : ndarray, shape (3, n+1)
        Normalised to 1.0 at the finest LH sub-band.
        ``ratios[0, i]`` / ``[1, i]`` / ``[2, i]`` are for LH / HL / HH
        at DWT level i+1.  ``ratios[0, n]`` covers both the coarsest LH
        detail band and the LL sub-band (they have nearly equal energies).
        Multiply by a scalar base_step to obtain a dwtstep array.
    """
    ratios = np.zeros((3, n + 1))
    m = H
    for i in range(n + 1):
        m //= 2
        positions = [
            (m // 2,     m + m // 2),   # LH
            (m + m // 2, m // 2),       # HL
            (m + m // 2, m + m // 2),   # HH
        ]
        for k, (r, c) in enumerate(positions):
            Z = np.zeros((H, H))
            Z[r, c] = 1.0
            Zr = nlevidwt(Z, n)
            ratios[k, i] = np.sqrt(np.sum(Zr ** 2))
    # LL sub-band: impulse at centre of deepest LP block [:m, :m]
    Z = np.zeros((H, H))
    Z[m // 2, m // 2] = 1.0
    Zr = nlevidwt(Z, n)
    ratios[0, n] = np.sqrt(np.sum(Zr ** 2))   # overwrites coarsest LH slot
    return ratios / ratios[0, 0]               # normalise to finest LH = 1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def dwtjpegenc(
    X: np.ndarray,
    dwtstep: np.ndarray,
    n: int = 4,
    opthuff: bool = False,
    dcbits: int = 9,
    log: bool = True,
) -> Tuple[np.ndarray, HuffmanTable]:
    """
    Encode a greyscale image using a multi-level DWT and JPEG Huffman coding.

    Transform chain:
        nlevdwt (n+1 levels)  ->  per-sub-band quant1  ->  dwtgroup
        ->  zig-zag scan  ->  run/amplitude coding  ->  Huffman encode

    Parameters
    ----------
    X : ndarray
        Zero-meaned greyscale input image (e.g. ``X = raw - 128``).
    dwtstep : ndarray, shape (3, n+1)
        Quantisation step sizes, one per sub-band level.
        ``dwtstep[0, i]`` / ``[1, i]`` / ``[2, i]`` are the LH / HL / HH
        steps at DWT level ``i+1``; ``dwtstep[0, n]`` is the step for
        the deepest LL sub-band.
        Use ``np.full((3, n+1), step)`` for equal-step quantisation.
    n : int
        Number of *extra* DWT levels applied after the first (matching the
        notebook convention).  Total levels = n+1; block size = 2^(n+1).
        Default 4 -> 5 DWT levels, 32x32 coding blocks.
    opthuff : bool
        If True, design custom Huffman tables from the image statistics.
    dcbits : int
        Fixed-length word size for the DC (LL) coefficient of each block.
    log : bool
        Print progress messages.

    Returns
    -------
    vlc : ndarray, shape (K, 2)
        Variable-length codewords.  ``vlc[:, 1].sum()`` is the total bit count.
    hufftab : HuffmanTable
        Huffman table used.  Pass to dwtjpegdec when opthuff=True.
    """
    if dwtstep.shape != (3, n + 1):
        raise ValueError(
            f'dwtstep must have shape (3, {n+1}) for n={n}, got {dwtstep.shape}')

    M = 2 ** (n + 1)  # coding block size

    # ---- n+1-level DWT (matches notebook nlevdwt convention) --------
    if log:
        print('Forward DWT: {} levels, block size {}x{}'.format(n + 1, M, M))
    Y = nlevdwt(X, n)

    # ---- per-sub-band quantisation to integer step indices ----------
    if log:
        print('Quantising sub-bands (dwtstep shape {})'.format(dwtstep.shape))
    Yq = quantise_subbands(Y, dwtstep, n)

    # ---- regroup into M x M coding blocks ---------------------------
    # dwtgroup expects the number of DWT levels = n+1
    Yg = dwtgroup(Yq.astype(float), n + 1)
    Yg = np.round(Yg).astype('int')

    # ---- zig-zag scan of AC coefficients ----------------------------
    scan = diagscan(M)

    # ---- first pass: default JPEG Huffman tables --------------------
    if log:
        print('Generating huffcode and ehuf using default tables')
    dhufftab = huffdflt(1)
    huffcode, ehuf = huffgen(dhufftab)

    sy = Yg.shape
    huffhist = np.zeros(16 ** 2)
    vlc = []
    if log:
        print('Coding rows')
    for r in range(0, sy[0], M):
        for c in range(0, sy[1], M):
            yqflat = Yg[r:r+M, c:c+M].flatten('F')
            dccoef = int(yqflat[0]) + 2 ** (dcbits - 1)
            if dccoef not in range(2 ** dcbits):
                raise ValueError(
                    'DC coefficient ({}) too large for dcbits={}'.format(
                        int(yqflat[0]), dcbits))
            vlc.append(np.array([[dccoef, dcbits]]))
            ra1 = runampl(yqflat[scan])
            vlc.append(huffenc(huffhist, ra1, ehuf))
    vlc = np.concatenate([np.zeros((0, 2), dtype=np.intp)] + vlc)

    if not opthuff:
        if log:
            print('Bits for coded image = {}'.format(vlc[:, 1].sum()))
        return vlc, dhufftab

    # ---- second pass: custom Huffman tables -------------------------
    if log:
        print('Generating huffcode and ehuf using custom tables')
    dhufftab = huffdes(huffhist)
    huffcode, ehuf = huffgen(dhufftab)

    if log:
        print('Coding rows (second pass)')
    huffhist = np.zeros(16 ** 2)
    vlc = []
    for r in range(0, sy[0], M):
        for c in range(0, sy[1], M):
            yqflat = Yg[r:r+M, c:c+M].flatten('F')
            dccoef = int(yqflat[0]) + 2 ** (dcbits - 1)
            vlc.append(np.array([[dccoef, dcbits]]))
            ra1 = runampl(yqflat[scan])
            vlc.append(huffenc(huffhist, ra1, ehuf))
    vlc = np.concatenate([np.zeros((0, 2), dtype=np.intp)] + vlc)

    if log:
        print('Bits for coded image = {}'.format(vlc[:, 1].sum()))
        print('Bits for huffman table = {}'.format(
            (16 + max(dhufftab.huffval.shape)) * 8))
    return vlc, dhufftab


def dwtjpegdec(
    vlc: np.ndarray,
    dwtstep: np.ndarray,
    n: int = 4,
    hufftab: Optional[HuffmanTable] = None,
    dcbits: int = 9,
    W: int = 256,
    H: int = 256,
    log: bool = True,
) -> np.ndarray:
    """
    Decode a DWT-JPEG bitstream back to a greyscale image.

    Inverse chain:
        Huffman decode  ->  dwtgroup^{-1}  ->  per-sub-band quant2
        ->  nlevidwt (n+1 levels)

    Parameters
    ----------
    vlc : ndarray, shape (K, 2)
        Variable-length codewords from dwtjpegenc.
    dwtstep : ndarray, shape (3, n+1)
        Quantisation step sizes — must match the encoder.
    n : int
        Number of extra DWT levels (must match the encoder).
    hufftab : HuffmanTable, optional
        Required when the encoder used opthuff=True.
    dcbits, W, H : int
        Must match the encoder / image size.
    log : bool
        Print progress messages.

    Returns
    -------
    Z : ndarray
        Reconstructed image (zero-meaned; add 128 to display).
    """
    if dwtstep.shape != (3, n + 1):
        raise ValueError(
            f'dwtstep must have shape (3, {n+1}) for n={n}, got {dwtstep.shape}')

    M = 2 ** (n + 1)
    opthuff = (hufftab is not None)
    scan = diagscan(M)

    if opthuff:
        if log:
            print('Generating huffcode and ehuf using custom tables')
    else:
        if log:
            print('Generating huffcode and ehuf using default tables')
        hufftab = huffdflt(1)

    huffstart = np.cumsum(np.block([0, hufftab.bits[:15]]))
    huffcode, ehuf = huffgen(hufftab)
    k = 2 ** np.arange(17)

    eob = ehuf[0]
    run16 = ehuf[15 * 16]
    i = 0
    Zq = np.zeros((H, W))

    if log:
        print('Decoding rows')
    for r in range(0, H, M):
        for c in range(0, W, M):
            yq = np.zeros(M ** 2)
            cf = 0

            if vlc[i, 1] != dcbits:
                raise ValueError(
                    'The bits for the DC coefficient does not agree with vlc table')
            yq[cf] = vlc[i, 0] - 2 ** (dcbits - 1)
            i += 1

            while np.any(vlc[i] != eob):
                run = 0
                while np.all(vlc[i] == run16):
                    run += 16
                    i += 1
                start = int(huffstart[vlc[i, 1] - 1])
                res = int(hufftab.huffval[start + vlc[i, 0] - huffcode[start]])
                run += res // 16
                cf += run + 1
                si = res % 16
                i += 1
                if vlc[i, 1] != si:
                    raise ValueError(
                        'Problem with decoding: wrong hufftab table?')
                ampl = int(vlc[i, 0])
                thr = int(k[si - 1])
                yq[scan[cf - 1]] = ampl - (ampl < thr) * (2 * thr - 1)
                i += 1
            i += 1

            Zq[r:r+M, c:c+M] = yq.reshape((M, M)).T

    # ---- undo dwtgroup ----------------------------------------------
    Zi_grouped = dwtgroup(Zq, -(n + 1))
    Zi_int = np.round(Zi_grouped).astype('int')

    # ---- per-sub-band dequantisation --------------------------------
    if log:
        print('Inverse quantising sub-bands')
    Zi = dequantise_subbands(Zi_int, dwtstep, n)

    # ---- n+1-level inverse DWT --------------------------------------
    if log:
        print('Inverse DWT: {} levels'.format(n + 1))
    Z = nlevidwt(Zi, n)

    return Z
