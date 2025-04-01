## ----------------------------------------------------------------------------
## Imports

import numpy as np
from numpy.typing import NDArray

csimap_thr = 1E-32

# --- csimap_eval_complex
#     evaluates the basic form for the CSI map and its derivative
def csimap_eval_complex(x  : NDArray,             # (num_px,)         | complex
                        te : NDArray,             # (num_te,)         | float
                        S  : NDArray,             # (num_px, num_te)  | complex
                        A  : NDArray              # (num_te, num_te)  | complex
                        ) -> NDArray:
  # phase
  P = np.outer(1j * x.real - x.imag, 2 * np.pi * te)
  # matrix
  return np.exp(P) * (A @ (np.exp(-P) * S).T).T

# --- csimap_eval_mtx_complex
#     evaluates the basic form for the matrix of the CSI map
def csimap_eval_mtx_complex(x  : NDArray,             # (num_px,)         | complex
                            te : NDArray,             # (num_te,)         | float
                            A  : NDArray              # (num_te, num_te)  | complex
                            ) -> NDArray:
  # phase
  p = np.kron(1j * x.real - x.imag, 2 * np.pi * te)
  # column scaling
  X = (np.tile(A, (1, x.size)) * np.exp(-p)).T.reshape((x.size, te.size, te.size)).transpose((0, 2, 1)).reshape((x.size * te.size, te.size))
  # row scaling
  return (X.T * np.exp(+p)).T.reshape((x.size, te.size, te.size))

# --- csimap_eval_complex_at_voxel
#     evaluates the basic form for the CSI map and its derivative at a voxel
def csimap_eval_complex_at_voxel(x  : complex,             # (1,)              | complex
                                 te : NDArray,             # (num_te,)         | float
                                 s  : NDArray,             # (num_te,)         | complex
                                 A  : NDArray              # (num_te, num_te)  | complex
                                ) -> NDArray:
  # phase
  p = te * (2 * np.pi * (1j * x.real - x.imag))
  # evaluate
  return np.exp(p) * (A @ (np.exp(-p) * s).T).T

# --- csimap_eval_jac_complex_at_voxel
#     evaluates the basic form for the jacobian of the CSI map at a voxel
def csimap_eval_mtx_complex_at_voxel(x  : complex,             # (1,)              | complex
                                     te : NDArray,             # (num_te,)         | float
                                     A  : NDArray              # (num_te, num_te)  | complex
                                    ) -> NDArray:
  # phase
  p = te * (2 * np.pi * (1j * x.real - x.imag))
  # evaluate
  return ((A * np.exp(-p)).T * np.exp(p)).T