## ----------------------------------------------------------------------------
## Imports

from math import prod
import numpy as np

from numpy.typing import NDArray

from CSITools.objects.csimap import CSIMap
from CSITools.routines.residual import csimap_residual_eval_complex, csimap_residual_eval_diff_complex

from CSITools.routines.gradients import assemble_gradient_matrices_2d

class CSIToolsSmooth(object):
  pass


# --- CSIToolsResidual
#     Implements half-norm-squared of the residual
class CSIToolsSumSquares(CSIToolsSmooth):
  def __init__(self, R : CSIMap):
    # set map
    self.map = R
    
  def eval(self, x : NDArray) -> float:
    return csimap_residual_eval_complex(x.ravel(), self.map.te, self.map.S, self.map.Pe)
    
  def grad(self, x : NDArray) -> NDArray:
    return csimap_residual_eval_diff_complex(x.ravel(), self.map.te, self.map.S, self.map.Pe, self.map.Pg)

# --- CSIToolsZero
#     Implements the smooth zero function
class CSIToolsZero(CSIToolsSmooth):
  def __init__(self):
    pass
    
  def eval(self, x : NDArray) -> float:
    return 0.0
    
  def grad(self, x : NDArray) -> NDArray:
    return np.zeros(x.shape, dtype=x.dtype)


# --- CSIToolsResidual
#     Implements half-norm-squared of the residual
class CSIToolsSmoothHuberTV(CSIToolsSmooth):
  def __init__(self, shape : tuple = None, t : float = 1.0, scale : float = 1.0, variable : str = 'fieldmap'):
    if variable not in [ 'f', 'field', 'fieldmap', 'r', 'r2', 'r2*', 'r2star' ]:
      raise ValueError("Variable must be either 'field' or 'r2*'.")
    if variable in [ 'f', 'field', 'fieldmap' ]:
      self.variable = 're'
    elif variable in [ 'r', 'r2', 'r2*', 'r2star' ]:
      self.variable = 'im'
    else:
      self.variable = 'both'
    # set shape
    self.shape = shape
    # dimension and number of pixels
    self.ndim = shape[-1]
    self.num_px = prod(shape) // shape[-1]
    self.im_shape = (shape[0], shape[1])
    # smoothing
    self.t = t
    # scaling 
    self.scale = scale
    # gradient matrices
    Gx, Gy = assemble_gradient_matrices_2d(self.im_shape)
    self.grad_x = Gx
    self.grad_y = Gy
    
  def eval(self, x : NDArray) -> float:
    if self.variable == 're':
      x = x.real
    if self.variable == 'im':
      x = x.imag
    # gradient
    g = np.vstack([ self.grad_x @ x.ravel(), self.grad_y @ x.ravel() ]).T
    # norm
    g_nrm = np.linalg.norm(g, ord=2, axis=1).ravel()
    return 0.5 * self.scale * np.sum(np.where(g_nrm <= self.t, g_nrm ** 2 / (2 * self.t), g_nrm - 0.5 * self.t))
    
  def grad(self, x : NDArray) -> NDArray:
    if self.variable == 're':
      x = x.real
    if self.variable == 'im':
      x = x.imag
    # gradient
    g = np.vstack([ self.grad_x @ x.ravel(), self.grad_y @ x.ravel() ]).T
    # shrinkage factor
    s = 1/np.maximum(self.t, np.linalg.norm(g, ord=2, axis=1).ravel())
    # scaled gradient
    gs = self.scale * (g.T * s).T
    gx = (self.grad_x.T @ gs.T[0] + self.grad_y.T @ gs.T[1])
    # return gradient
    if self.variable == 're':
      return gx.astype('complex').reshape(x.shape)
    if self.variable == 'im':
      return 1j * gx.astype('complex').reshape(x.shape)
    return gx.reshape(x.shape)