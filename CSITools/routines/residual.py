## ----------------------------------------------------------------------------
## Imports

import numpy as np
from numpy.typing import NDArray

from CSITools.routines.csimap import csimap_eval_complex, csimap_eval_complex_at_voxel

# --- csimap_residual_eval_complex
#     evaluates the basic form for the CSI map residual
def csimap_residual_eval_complex(x  : NDArray,             # (num_px,)         | complex
                                 te : NDArray,             # (num_te,)         | float
                                 S  : NDArray,             # (num_px, num_te)  | complex
                                 A  : NDArray              # (num_te, num_te)  | complex
                                ) -> NDArray:              # (num_px,)         | float
  # evaluate
  return 0.5 * np.sum(np.linalg.norm(csimap_eval_complex(x, te, S, A), axis=1) ** 2)

# --- csimap_residual_eval_diff_complex
#     evaluates the basic form for the derivative of the CSI map residual
def csimap_residual_eval_diff_complex(x  : NDArray,             # (num_px,)         | complex
                                      te : NDArray,             # (num_te,)         | float
                                      S  : NDArray,             # (num_px, num_te)  | complex
                                      A  : NDArray,             # (num_te, num_te)  | complex
                                      DA : NDArray              # (num_te, num_te)  | complex
                                      ) -> NDArray:             # (num_px,)         | complex
  # evaluate
  return 0.5 * np.sum(np.conj(csimap_eval_complex(x, te, S, A)) * csimap_eval_complex(x, te, S, DA), axis=1).ravel()

# --- csimap_residual_eval_complex_at_voxel
#     evaluates the basic form for the CSI map residual at a voxel
def csimap_residual_eval_complex_at_voxel(x  : complex,             # (1,)              | complex
                                          te : NDArray,             # (num_te,)         | float
                                          s  : NDArray,             # (num_te,)         | complex
                                          A  : NDArray              # (num_te, num_te)  | complex
                                          ) -> NDArray:              # (num_px,)         | float
  # evaluate
  return 0.5 * np.linalg.norm(csimap_eval_complex_at_voxel(x, te, s, A).ravel())

# --- csimap_residual_eval_diff_complex
#     evaluates the basic form for the derivative of the CSI map residual at a voxel
def csimap_residual_eval_diff_complex_at_voxel(x  : NDArray,             # (num_px,)         | complex
                                               te : NDArray,             # (num_te,)         | float
                                               s  : NDArray,             # (num_te,)         | complex
                                               A  : NDArray,             # (num_te, num_te)  | complex
                                               DA : NDArray              # (num_te, num_te)  | complex
                                               ) -> NDArray:             # (num_px,)         | complex
  # evaluate
  return 0.5 * np.sum(np.conj(csimap_eval_complex_at_voxel(x, te, s, A)) * csimap_eval_complex_at_voxel(x, te, s, DA), axis=1).ravel()