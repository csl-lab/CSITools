## ----------------------------------------------------------------------------
## Imports

import numpy as np
from numpy.typing import NDArray

# --- simulate signal
#     simulates the signal predicted by the given parameters
def simulate_signal(M     : NDArray,              # (num_te, num_sp)  | complex
                    te    : NDArray,              # (num_te,)         | float
                    c     : NDArray,              # (num_vx, num_sp)  | comples
                    x     : NDArray,              # (num_vx,)         | complex
                    ) -> NDArray:
  # validate
  if x.ndim not in [ 1, 2, 3 ]:
    raise ValueError('Only 1, 2 or 3 spatial dimensions are supported')
  # number of echo times
  num_te = M.shape[0]
  # number of species
  num_sp = M.shape[1]
  # number of voxels
  num_vx = x.size
  # shape
  shape = x.shape
  # validation
  if te.size != num_te:
      raise ValueError('The number of rows of the model matrix must match the number of echo times.')
  if c.shape[-1] != num_sp:
      raise ValueError('The number of columns of the model matrix must match the number of species in the concentration vector.')
  if c.size != num_vx * num_sp:
      raise ValueError('The number of elements in the concentration vector must match the number of species and number of voxels.')
  # evaluate
  return (np.exp(np.outer(2 * np.pi * 1j * te, x)) * (M @ c.reshape((num_vx, num_sp)).T)).T.reshape(shape + (num_te,))

# --- get concentration
#    gets concentrations from parameters
def get_concentration(Mp    : NDArray,              # (num_te, num_sp)  | complex
                      te    : NDArray,              # (num_te,)         | float
                      S     : NDArray,              # (num_vx, )
                      x     : NDArray,              # (num_vx,)         | complex
                      ) -> NDArray:
  # validate
  if x.ndim not in [ 1, 2, 3 ]:
    raise ValueError('Only 1, 2 or 3 spatial dimensions are supported')
  # number of echo times
  num_te = Mp.shape[1]
  # number of species
  num_sp = Mp.shape[0]
  # shape
  shape = x.shape
  # number of voxels
  num_vx = x.size
  # evaluate and return
  return (Mp @ (np.exp(-np.outer(x, 2 * np.pi * 1j * te)) * S.reshape((num_vx, num_te))).T).T.reshape(shape + (num_sp,))