# KitNET (vendored)

Source: https://github.com/ymirsky/KitNET-py (MIT License, see `LICENSE.txt`).
Author: Yisroel Mirsky.

Vendored verbatim (`KitNET.py`, `dA.py`, `corClust.py`, `utils.py`) because
these files use old-style flat imports (`import dA as AE`, `from utils
import *`) that assume same-directory placement, not a Python package --
`src/models/kitsune_real_kitnet.py` adds this directory to `sys.path`
before importing rather than rewriting the authors' code.

This is the REAL KitNET algorithm (online, incremental ensemble of small
autoencoders + an output autoencoder), used to check whether xNIDS's own
`Scripts/kitsune.py` (a single monolithic Keras autoencoder, batch-trained)
is actually what produced the paper's Table 12 Kitsune numbers, or whether
Table 12 reflects this original algorithm instead -- see
Appendix C.1 of the xNIDS paper, which specifies `m=10` (max autoencoder
size) as the one Kitsune hyperparameter they report.
