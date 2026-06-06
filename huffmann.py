"""
JPEG-style Huffman encoder/decoder for DWT sub-band images.

The caller handles the DWT transform and per-sub-band quantisation.
This module takes the resulting integer sub-band array and produces a
compact bitstream using run-length + Huffman coding (JPEG AC coding scheme).

Usage
-----
    vlc, hufftab = huffman_encode(Yq, n, opthuff=True, dcbits=11)
    Yq_dec       = huffman_decode(vlc, n, hufftab=hufftab, dcbits=11)

Yq  — (H, W) integer array from per-sub-band quant1 of nlevdwt output
n   — extra DWT levels; block size M = 2^(n+1)  (e.g. n=3 → 16×16 blocks)
vlc — (K, 2) array of (codeword, bitlength); vlc[:, 1].sum() = total bits
"""
import numpy as np
from cued_sf2_lab.jpeg import (
    HuffmanTable, diagscan, dwtgroup,
    huffdes, huffdflt, huffenc, huffgen, runampl,
)


def huffman_encode(
    Yq: np.ndarray,
    n: int,
    opthuff: bool = False,
    dcbits: int = 9,
) -> tuple:
    """
    Encode a quantised DWT sub-band image to a Huffman bitstream.

    Parameters
    ----------
    Yq      : (H, W) integer array — output of per-sub-band quant1
    n       : extra DWT levels; block size M = 2^(n+1)
    opthuff : design custom Huffman tables from image statistics (two-pass)
    dcbits  : fixed word length for the DC (LL) coefficient
              Use dcbits=11 when LL step is small (inverse-MSE scheme)

    Returns
    -------
    vlc     : (K, 2) array; vlc[:, 1].sum() gives total compressed bits
    hufftab : Huffman table used — pass to huffman_decode when opthuff=True
    """
    M = 2 ** (n + 1)
    Yg = np.round(dwtgroup(Yq.astype(float), n + 1)).astype(int)
    scan = diagscan(M)
    huffhist = np.zeros(16 ** 2)

    def encode_blocks(ehuf):
        result = []
        for r in range(0, Yg.shape[0], M):
            for c in range(0, Yg.shape[1], M):
                flat = Yg[r:r+M, c:c+M].flatten('F')
                dc = int(flat[0]) + 2 ** (dcbits - 1)
                if dc not in range(2 ** dcbits):
                    raise ValueError(
                        f'DC={int(flat[0])} overflows {dcbits}-bit range; '
                        f'increase dcbits (currently {dcbits})')
                result.append(np.array([[dc, dcbits]]))
                result.append(huffenc(huffhist, runampl(flat[scan]), ehuf))
        return np.concatenate([np.zeros((0, 2), dtype=np.intp)] + result)

    hufftab = huffdflt(1)
    _, ehuf = huffgen(hufftab)
    vlc = encode_blocks(ehuf)

    if opthuff:
        hufftab = huffdes(huffhist)
        _, ehuf = huffgen(hufftab)
        huffhist[:] = 0
        vlc = encode_blocks(ehuf)

    print(f'Bits: {int(vlc[:, 1].sum()):,}')
    return vlc, hufftab


def huffman_decode(
    vlc: np.ndarray,
    n: int,
    hufftab: HuffmanTable = None,
    dcbits: int = 9,
    W: int = 256,
    H: int = 256,
) -> np.ndarray:
    """
    Decode a Huffman bitstream back to a quantised DWT sub-band image.

    Parameters
    ----------
    vlc     : (K, 2) array from huffman_encode
    n       : extra DWT levels — must match encoder
    hufftab : custom table when opthuff=True; None uses default JPEG tables
    dcbits  : must match encoder
    W, H    : image dimensions

    Returns
    -------
    Yq : (H, W) integer sub-band image
         Apply dequantise_subbands + nlevidwt to reconstruct the image.
    """
    M = 2 ** (n + 1)
    if hufftab is None:
        hufftab = huffdflt(1)

    huffstart = np.cumsum(np.block([0, hufftab.bits[:15]]))
    huffcode, _ = huffgen(hufftab)
    _, ehuf = huffgen(hufftab)
    k = 2 ** np.arange(17)
    eob = ehuf[0]
    run16 = ehuf[15 * 16]
    scan = diagscan(M)

    i = 0
    Zq = np.zeros((H, W))
    for r in range(0, H, M):
        for c in range(0, W, M):
            yq = np.zeros(M ** 2)
            cf = 0
            if vlc[i, 1] != dcbits:
                raise ValueError('DC bits mismatch — wrong dcbits or hufftab?')
            yq[0] = vlc[i, 0] - 2 ** (dcbits - 1)
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
                    raise ValueError('Amplitude bits mismatch — wrong hufftab?')
                ampl = int(vlc[i, 0])
                thr = int(k[si - 1])
                yq[scan[cf - 1]] = ampl - (ampl < thr) * (2 * thr - 1)
                i += 1
            i += 1

            Zq[r:r+M, c:c+M] = yq.reshape((M, M)).T

    return np.round(dwtgroup(Zq, -(n + 1))).astype(int)
