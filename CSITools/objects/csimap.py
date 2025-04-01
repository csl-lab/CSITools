## ----------------------------------------------------------------------------
## Imports

import numpy as np
from numpy.typing import NDArray

from CSITools.routines.csimap import csimap_eval_complex, csimap_eval_mtx_complex
from CSITools.routines.models import get_wf_model, assemble_model_matrix
from CSITools.routines.signal import simulate_signal, get_concentration

class CSIMap(object):
  def __init__(self, 
               signal   : NDArray,
               te       : NDArray,
               shifts   : list     = None,
               weights  : list     = None,
               field    : float    = None,
               model    : NDArray  = None,
               stol     : float    = 0.0
               ):
    # number of echo times
    num_te = te.size
    if shifts is None or weights is None:
      if model is None:
        UserWarning('No chemical shifts and weights nor model matrix was provided. Using default multipeak water and fat model with 1.5T.')
        # set field
        field = 1.5
        # get default model
        shifts, weights = get_wf_model(field, mp=True)
        # model matrix
        M = assemble_model_matrix(te, shifts, weights)
        # number of species
        num_sp = 2
      else:
        if model.shape[0] != te.size:
          raise ValueError('The number of rows of the model matrix and the number of echo times must be the same.')
        # model matrix
        M = model
        # number of species
        num_sp = M.shape[1]
    else:
      # model matrix
      M = assemble_model_matrix(te, shifts, weights)
      # number of species
      num_sp = M.shape[1]
    if signal.shape[-1] != te.size:
      raise ValueError('The number of signal acquisitions must be the same as the number of echo times.')
    if signal.ndim not in [ 2, 3, 4 ]:
      raise ValueError('Only 1D, 2D or 3D arrays are allowed.')
    # image shape
    im_shape = signal.shape[:signal.ndim-1]
    # number of pixels
    num_px = np.prod(im_shape)
    # pseudoinverse
    Mp = np.linalg.pinv(M)
    # projector / eval
    Pe = np.eye(num_te) - M @ Mp
    # commutator / derivative
    Pd = 2 * np.pi * 1j * (np.diag(te) @ Pe - Pe @ np.diag(te))
    # assign fields
    # - signal
    self.S = signal.reshape((num_px, num_te))
    # - thresholding
    idx_S = np.where(np.linalg.norm(self.S, axis=1) <= stol)[0]
    self.S[idx_S] = np.zeros((idx_S.size, num_te))
    # - echo times
    self.te = te
    self.num_te = num_te
    # - species
    self.shifts = shifts
    self.weights = weights
    self.field = field
    self.num_sp = num_sp
    # - image
    self.im_shape = im_shape
    self.num_px = num_px
    # - model
    self.M = M
    self.Mp = Mp
    self.Pe = Pe
    self.Pg = Pd

  def eval(self, 
           x : NDArray
           ) -> NDArray:
    if x.ndim == 1:
      return csimap_eval_complex(x, self.te, self.S, self.Pe)    
    return csimap_eval_complex(x.ravel(), self.te, self.S, self.Pe).reshape(self.signal.shape)
  
  def grad(self, 
           x : NDArray
           ) -> NDArray:
    if x.ndim == 1:
      return csimap_eval_complex(x, self.te, self.S, self.Pg)    
    return csimap_eval_complex(x.ravel(), self.te, self.S, self.Pg).reshape(self.signal.shape)
  
  def mxt(self, 
           x : NDArray
           ) -> NDArray:
    if x.ndim == 1:
      return csimap_eval_mtx_complex(x, self.te,self.Pg)    
    return csimap_eval_mtx_complex(x.ravel(), self.te, self.Pg).reshape(self.signal.shape)
  
  def jac(self, 
           x : NDArray
           ) -> NDArray:
    if x.ndim == 1:
      return csimap_eval_mtx_complex(x, self.te, self.Pg)    
    return csimap_eval_mtx_complex(x.ravel(), self.te, self.Pg).reshape(self.signal.shape)
  
  def concentration(self,
                    x : NDArray,
                    S : NDArray | None = None
                    ) -> NDArray:
    if S is None:
      return get_concentration(self.Mp, self.te, self.S, x)
    return get_concentration(self.Mp, self.te, S, x)

  def signal(self,
             x : NDArray,
             c : NDArray | None = None,
             ) -> NDArray:
    if c is None:
      return simulate_signal(self.M, self.te, self.concentration(x), x)
    return simulate_signal(self.M, self.te, c, x)
