## ----------------------------------------------------------------------------
## Imports

import numpy as np
import cvxpy as cp

from numpy.typing import NDArray

from CSITools.routines.gradients import assemble_gradient_matrices_2d

class CSIToolsProx(object):
  pass

class CSIToolsProxPosImag(CSIToolsProx):
  def __init__(self, shape : tuple, eps : float = 1E-12):
    # input shape
    self.shape = shape
    # tolerance
    self.eps = eps

  def eval(self, x : NDArray) -> float:
    # return value
    if np.any(x.imag <= self.eps):
      return 0.0
    return 0.0

  def prox(self, x : NDArray, t : float) -> NDArray:
    # return
    return np.where(x.imag <= self.eps, x.real.astype(complex), x)

class CSIToolsProxTV(CSIToolsProx):
  def __init__(self, shape : tuple, scale : float = 1.0, variable : str = 'field'):
    if variable not in [ 'f', 'field', 'fieldmap', 'r', 'r2', 'r2*', 'r2star' ]:
      raise ValueError("Variable must be either 'field' or 'r2*'.")
    if variable in [ 'f', 'field', 'fieldmap' ]:
      self.variable = 're'
    else:
      self.variable = 'im'
    # input shape
    self.shape = shape
    # scale
    self.scale = scale
    # cvxpy variables
    # - parameters
    self.cp_t = cp.Parameter((1,), pos=True)
    self.cp_t.value = np.ones((1,), dtype=float)
    self.cp_xo = cp.Parameter(shape)
    # - variables
    if variable == 're':
      self.cp_x = cp.Variable(shape)
    else:
      self.cp_x = cp.Variable(shape, nonneg=True)

    # - objective
    self.cp_obj = cp.Minimize(0.5 * cp.sum_squares(self.cp_x - self.cp_xo) + self.cp_t * cp.tv(self.cp_x))
    # - problem
    self.cp_prb = cp.Problem(self.cp_obj)
    # - options
    self.options = [  
                      ('CLARABEL', { 'max_iter'               : 100000, 
                                     'reduced_tol_gap_abs'    : 1E-4, 
                                     'reduced_tol_gap_rel'    : 5E-5,
                                     'reduced_tol_feas'       : 1E-4,
                                     'reduced_tol_infeas_abs' : 1E-4,
                                     'reduced_tol_infeas_rel' : 5E-5,
                                     'reduced_tol_ktratio'    : 1E-4 
                                     }),
                      ('OSQP', { 'max_iter' : 50000, 
                                 'eps_abs'  : 1E-4, 
                                 'eps_rel'  : 1E-5 
                                }),
                      ('SCS',  { 'max_iters' : 10000, 
                                 'eps'       : 1E-5 
                                }) 
                    ]

  def eval(self, x : NDArray, eps : float = 1E-12) -> float:
    # assign to parameter
    if self.variable == 're':
      self.cp_xo.value = x.real.reshape(self.shape)
    else:
      if np.any(x.imag) <= eps:
        return np.inf
      self.cp_xo.value = x.imag.reshape(self.shape)
    # return value
    return self.scale * cp.tv(self.cp_xo).value

  def prox(self, x : NDArray, t : float, eps : float = 1E-12) -> NDArray:
    # update parameter values
    self.cp_t.value[0] = self.scale * t
    if self.variable == 're':
      self.cp_xo.value = x.real.reshape(self.shape)
    else:
      self.cp_xo.value = x.imag.reshape(self.shape)
    # solve
    self.cp_prb.solve(ignore_dpp=True, warm_start=True, verbose=False, solver_verbose=False, solver='CLARABEL')
    # return
    if self.variable == 're':
      return (self.cp_x.value.astype('complex').ravel() + 1j * np.maximum(0.0, x.imag).astype('complex')).reshape(x.shape)
    return (x.real.astype('complex').ravel() + 1j * self.cp_x.value.astype('complex').ravel()).reshape(x.shape)
  
class CSIToolsProxGradientBound(CSIToolsProx):
  def __init__(self, shape : tuple, gbound : float | NDArray = 1.0):
    # input shape
    self.shape = shape
    # number of pixels
    self.num_px = shape[0] * shape[1]
    # number of pixels with gradients
    self.num_grad_px = (shape[0] - 1) * (shape[1] - 1)
    if isinstance(gbound, float):
      self.gbound = gbound * np.ones((self.num_grad_px,), dtype=float)
    else:
      self.gbound = gbound.ravel()
    # cvxpy variables
    # - parameters
    self.cp_xo = cp.Parameter((self.num_px,))
    # - variables
    self.cp_x = cp.Variable((self.num_px,))
    # - objective
    self.cp_obj = cp.Minimize(0.5 * cp.sum_squares(self.cp_x - self.cp_xo))
    # - constraints
    Gx, Gy = assemble_gradient_matrices_2d(shape)
    self.grad_x = Gx
    self.grad_y = Gy
    self.cp_cnt = [ cp.norm(cp.vstack([ self.grad_x @ self.cp_x, self.grad_y @ self.cp_x ]), p=2, axis=0) <= self.gbound ] 
    # - problem
    self.cp_prb = cp.Problem(self.cp_obj, self.cp_cnt)
    # - options
    self.options = [  
                      ('CLARABEL', { 'max_iter'               : 100000, 
                                     'reduced_tol_gap_abs'    : 1E-4, 
                                     'reduced_tol_gap_rel'    : 5E-5,
                                     'reduced_tol_feas'       : 1E-4,
                                     'reduced_tol_infeas_abs' : 1E-4,
                                     'reduced_tol_infeas_rel' : 5E-5,
                                     'reduced_tol_ktratio'    : 1E-4 
                                     }),
                      ('OSQP', { 'max_iter' : 50000, 
                                 'eps_abs'  : 1E-4, 
                                 'eps_rel'  : 1E-5 
                                }),
                      ('SCS',  { 'max_iters' : 10000, 
                                 'eps'       : 1E-5 
                                }) 
                    ]

  def eval(self, x : NDArray, eps : float = 1E-2) -> float:
    # assemble gradient
    gx_nrm = np.linalg.norm(np.vstack([ self.grad_x @ x.real.ravel(), self.grad_y @ x.real.ravel() ]), axis=0)
    # return value
    if np.any(self.gbound < gx_nrm - eps):
      return np.inf
    return 0.0

  def prox(self, x : NDArray, t : float, eps : float = 1E-12) -> NDArray:
    # update parameter values
    self.cp_xo.value = x.real.ravel()
    # solve
    self.cp_prb.solve(ignore_dpp=True, warm_start=True, verbose=False, solver_verbose=False, solver='CLARABEL')
    # return
    return (self.cp_x.value.astype('complex').ravel() + 1j * np.maximum(0.0, x.imag.ravel()).astype('complex')).reshape(x.shape)

class CSIToolsProxLaplacianBound(CSIToolsProx):
  def __init__(self, shape : tuple, lbound : float | NDArray = 1.0):
    # input shape
    self.shape = shape
    # number of pixels
    self.num_px = shape[0] * shape[1]
    if isinstance(lbound, float):
      self.lbound = lbound * np.ones((self.num_px,), dtype=float)
    else:
      self.lbound = lbound.ravel()
    # cvxpy variables
    # - parameters
    self.cp_xo = cp.Parameter((self.num_px,))
    # - variables
    self.cp_x = cp.Variable((self.num_px,))
    # - objective
    self.cp_obj = cp.Minimize(0.5 * cp.sum_squares(self.cp_x - self.cp_xo))
    # - constraints
    Gx, Gy = assemble_gradient_matrices_2d(shape)
    self.laplacian = Gx.T @ Gx + Gy.T @ Gy
    self.cp_cnt = [ cp.abs(self.laplacian @ self.cp_x) <= self.lbound ] 
    # - problem
    self.cp_prb = cp.Problem(self.cp_obj, self.cp_cnt)
    # - options
    self.options = [  
                      ('CLARABEL', { 'max_iter'               : 100000, 
                                     'reduced_tol_gap_abs'    : 1E-4, 
                                     'reduced_tol_gap_rel'    : 5E-5,
                                     'reduced_tol_feas'       : 1E-4,
                                     'reduced_tol_infeas_abs' : 1E-4,
                                     'reduced_tol_infeas_rel' : 5E-5,
                                     'reduced_tol_ktratio'    : 1E-4 
                                     }),
                      ('OSQP', { 'max_iter' : 50000, 
                                 'eps_abs'  : 1E-4, 
                                 'eps_rel'  : 1E-5 
                                }),
                      ('SCS',  { 'max_iters' : 10000, 
                                 'eps'       : 1E-5 
                                }) 
                    ]
    
  def eval(self, x : NDArray, eps : float = 1E-2) -> float:
    # assemble laplacian
    abs_lapl_x = self.laplacian @ x.real.ravel()
    # return value
    if np.any(self.lbound < abs_lapl_x - eps):
      return np.inf
    return 0.0

  def prox(self, x : NDArray, t : float, eps : float = 1E-12) -> NDArray:
    # update parameter values
    self.cp_xo.value = x.real.ravel()
    # solve
    self.cp_prb.solve(ignore_dpp=True, warm_start=True, verbose=False, solver_verbose=False, solver='CLARABEL')
    if self.cp_x.value is None:
      self.cp_prb.solve(ignore_dpp=True, warm_start=True, verbose=True, solver_verbose=False, solver='CLARABEL')
    # return
    return (self.cp_x.value.astype('complex').ravel() + 1j * np.maximum(0.0, x.imag.ravel()).astype('complex')).reshape(x.shape)

class CSIToolsProxCSITools(CSIToolsProx):
  def __init__(self, shape : tuple, scale : float = 1.0, gbound : float | NDArray = 1.0):
    # scale
    self.scale = scale
    # input shape
    self.shape = shape
    # number of pixels
    self.num_px = shape[0] * shape[1]
    # number of pixels with gradients
    self.num_grad_px = (shape[0] - 1) * (shape[1] - 1)
    # gradient
    Gx, Gy = assemble_gradient_matrices_2d(shape)
    self.grad_x = Gx
    self.grad_y = Gy
    # bound on gradient norm
    if isinstance(gbound, float):
      self.gbound = gbound * np.ones((self.num_grad_px,), dtype=float)
    else:
      self.gbound = gbound.ravel()
    # cvxpy for fieldmap
    # - parameters
    self.cp_fo = cp.Parameter((self.num_px,))
    # - variables
    self.cp_f = cp.Variable((self.num_px,))
    # - objective
    self.cp_obj_f = cp.Minimize(0.5 * cp.sum_squares(self.cp_f - self.cp_fo))
    # - constraints
    self.cp_cnt_f = [ cp.norm(cp.vstack([ self.grad_x @ self.cp_f, self.grad_y @ self.cp_f ]), p=2, axis=0) <= self.gbound ] 
    # - problem
    self.cp_prb_f = cp.Problem(self.cp_obj_f, self.cp_cnt_f)
    # cvxpy for r2*
    # - parameters
    self.cp_t_r = cp.Parameter((1,), pos=True)
    self.cp_t_r.value = np.ones((1,), dtype=float)
    self.cp_ro = cp.Parameter(shape)
    # - variables
    self.cp_r = cp.Variable(shape, nonneg=True)
    # - objective
    self.cp_obj_r = cp.Minimize(0.5 * cp.sum_squares(self.cp_r - self.cp_ro) + self.cp_t_r * cp.tv(self.cp_r))
    # - problem
    self.cp_prb_r = cp.Problem(self.cp_obj_r)
    # - options
    self.options = [  
                      ('CLARABEL', { 'max_iter'               : 100000, 
                                     'reduced_tol_gap_abs'    : 1E-4, 
                                     'reduced_tol_gap_rel'    : 5E-5,
                                     'reduced_tol_feas'       : 1E-4,
                                     'reduced_tol_infeas_abs' : 1E-4,
                                     'reduced_tol_infeas_rel' : 5E-5,
                                     'reduced_tol_ktratio'    : 1E-4 
                                     }),
                      ('OSQP', { 'max_iter' : 50000, 
                                 'eps_abs'  : 1E-4, 
                                 'eps_rel'  : 1E-5 
                                }),
                      ('SCS',  { 'max_iters' : 10000, 
                                 'eps'       : 1E-5 
                                }) 
                    ]

  def eval(self, x : NDArray, eps : float = 1E-2) -> float:
    # assemble gradient
    gx_nrm = np.linalg.norm(np.vstack([ self.grad_x @ x.real.ravel(), self.grad_y @ x.real.ravel() ]), axis=0)
    # return value
    if np.any(self.gbound < gx_nrm - eps):
      return np.inf
      # assign to parameter
    if np.any(x.imag) <= 1E-12:
      return np.inf
    self.cp_ro.value = x.imag.reshape(self.shape)
    # return value
    return self.scale * cp.tv(self.cp_ro).value

  def prox(self, x : NDArray, t : float) -> NDArray:
    # update parameter values
    self.cp_fo.value = x.real.ravel()
    self.cp_ro.value = x.imag.reshape(self.shape)
    # solve for f
    self.cp_prb_f.solve(ignore_dpp=True, warm_start=True, verbose=False, solver_verbose=False, solver='CLARABEL')
    # solve for r
    self.cp_prb_r.solve(ignore_dpp=True, warm_start=True, verbose=False, solver_verbose=False, solver='CLARABEL')
    # return
    return (self.cp_f.value.astype('complex').ravel() + 1j * self.cp_r.value.astype('complex').ravel()).reshape(x.shape)