## ----------------------------------------------------------------------------
## Imports

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import spmatrix

from scipy.sparse import csr_array

# .ravel() unrolls rows first

# --- assemble_gradient_matrices_2d
#     assembles matrices to compute partial derivatives in 2D
def assemble_gradient_matrices_2d(shape : tuple) -> tuple[spmatrix]:
  # number of pixels
  num_px = shape[0] * shape[1]
  # number of pixels with gradient
  num_grad_px = (shape[0] - 1)* (shape[1] - 1)
  # indices of pixels with gradients
  idx = np.linspace(0, num_px - 1, num_px).astype(int)
  idx = idx[np.logical_and((idx + 1) % shape[1] != 0, idx // shape[1] < shape[0] - 1)]

  # assemble matrices in CSR sparse format
  # - values
  val = -(-1) ** np.linspace(0, 2 * num_grad_px - 1, 2 * num_grad_px, endpoint=True).astype(int)
  # - gradient in x
  ind_x = np.vstack([ idx, idx + shape[1] ]).T.ravel()
  ind_x_ptr = 2 * np.linspace(0, num_grad_px, num_grad_px + 1).astype(int)

  # print(val, ind_x, ind_x_ptr)

  grad_x = csr_array((val, ind_x, ind_x_ptr), shape=(num_grad_px , num_px))

  # - gradient in y
  ind_y = np.vstack([ idx, idx + 1 ]).T.ravel()
  ind_y_ptr = 2 * np.linspace(0, num_grad_px, num_grad_px + 1).astype(int)

  # print(val, ind_y, ind_y_ptr)

  grad_y = csr_array((val, ind_y, ind_y_ptr), shape=(num_grad_px , num_px))
  
  return grad_x, grad_y