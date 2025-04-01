## ----------------------------------------------------------------------------
## Imports

import numpy as np
from numpy.typing import NDArray

from CSITools.objects.solver import CSIToolsSolver

from CSITools.objects.csimap import CSIMap
from CSITools.objects.smooth import CSIToolsSumSquares
from CSITools.objects.prox import CSIToolsProx

from CSITools.objects.prox import CSIToolsProxPosImag

# --- CSIToolsSolver
#     Solves non-linear least-squares
#     No regularization
#     The imaginary part of the variable must be non-negative
class CSIToolsSolverPGD(CSIToolsSolver):
  def __init__(self, R : CSIMap):
    # the csi map
    self.map = R
    # the residual
    self.objective = CSIToolsSumSquares(R)
    # options
    self.options = {
                      'max_itns'           : 10000,
                      'stepAbsTol'         : 1E-4,
                      'stepRelTol'         : 1E-6,
                      'gradAbsTol'         : 1E-4,
                      'gradRelTol'         : 1E-6,
                      'linesearch'         : True,
                      'ls_max_itns'        : 1000,
                      'ls_fwd_factor'      : 1.00,
                      'ls_fwd_every'       : 1000,
                      'ls_bck_factor'      : 0.95,
                      'verbose'            : True,
                      'print_every'        : 100,
                      'print_header_every' : 10
                  }
    self.termination = [
                          'Reached maximum number of iterations',
                          'Reached absolute tolerance for step',
                          'Reached relative tolerance for step'
                      ]
    # logging
    self.log = {
                  'itns' : 0
              }
    # last computed iterate
    self.xn = None
    # last solution
    self.x = None
    # threshold
    self.eps = 1E-32

  def solve(self, xo : NDArray | None = None, t : float = 1.0, prox : CSIToolsProx | None = None, mode : str = 'both') -> NDArray:
    # initialize
    if xo is None:
      x = np.zeros((self.map.num_px,), dtype=complex)
    else:
      if xo.size != self.map.num_px:
        raise ValueError('The input vector must have shape ', self.map.im_shape, ' or length ', self.map.num_px)
      x = np.where(xo < 0.0, xo.real, xo).ravel()
    if not x.dtype == 'complex':
      x = x.astype('complex')
    # initialize prox
    if prox is None:
      prox = CSIToolsProxPosImag(shape=(self.map.num_px,))
    # options
    opts = self.options
    # projectors
    if mode in [ 'f', 'field', 'fieldmap' ]:
      proj_grad = lambda g : g.real.astype('complex')
    elif mode in [ 'r', 'r2', 'r2*', 'r2star' ]:
      proj_grad = lambda g : 1j * g.imag.astype('complex')
    elif mode in [ 'both' ]:
      proj_grad = lambda g : g
    else:
      raise ValueError('Unknown mode.')
    # clear log
    # TODO
    # initialize
    itn = 0
    itn_hdr = 0
    ls_itns = 0
    ls_fwd_every = 0
    # main loop
    stop = False
    if opts['verbose']:
      # setup strings
      header_str = f"{'itn':8s}| {'obj(x)':9s} | {'|x|':9s} | {'|g|':9s} | {'|dx|':9s} | {'t':9s} | {'ls itn':6s} | {'min(r)':9s} | {'max(r)':9s}"
      # header
      line_str = '%07d | %.3E | %.3E | %.3E | %.3E | %.3E | %6d | %.3E | %.3E'
      line_str_inf = '%07d | %9s | %.3E | %.3E | %.3E | %.3E | %6d | %.3E | %.3E'
    while not stop:
      itn += 1
      # norm of iterate
      x_nrm = np.linalg.norm(x)
      # current objective
      obj_x = self.objective.eval(x) + prox.eval(x)
      # descent direction
      # - gradient
      g = self.objective.grad(x)
      # - steepest ascent direction is the conjugate
      g = np.conj(g) 
      # - project
      g = proj_grad(g) 
      # - norm
      g_nrm = np.linalg.norm(g.ravel())
      # linesearch
      if opts['linesearch']:
        # forward step
        if ls_fwd_every % opts['ls_fwd_every'] == 0:
          t = opts['ls_fwd_factor'] * t
        ls_fwd_every = 1
        # initialize
        ls_itns = 0
        # trial
        # - step
        xt = x - t * g
        # - proximal step
        xp = prox.prox(xt, t)
        # - distance
        dist_xt = np.linalg.norm(xt - xp) ** 2
        # - objective
        obj_xp = self.objective.eval(xp) + prox.eval(xp)
        # linesearch
        ls_stop = False
        while not ls_stop:
          if ls_itns == opts['ls_max_itns'] or obj_xp <= obj_x - 0.5 * g_nrm ** 2 / t + 0.5 * t * dist_xt ** 2:
            ls_stop = True
          else:
            ls_itns += 1
            # backward 
            t = t * opts['ls_bck_factor']
            # trial
            # - step
            xt = x - t * g
            # - proximal step
            xp = prox.prox(xt, t)
            # - distance
            dist_xt = np.linalg.norm(xt - xp) ** 2
            # - objective
            obj_xp = self.objective.eval(xp) + prox.eval(xp)
      else:
        # step
        # - step
        xt = x - t * g
        # - proximal step
        xp = prox.prox(xt, t)
        # - objective
        obj_xp = self.objective.eval(xp) + prox.eval(xp)
      # update
      # - norm of step
      dx_nrm = np.linalg.norm(xp - x)
      # - step
      x = xp
      # log
      self.xn = xp
      # stopping criteria
      stop_criteria = [
                        itn == opts['max_itns'],              # maximum iterations
                        dx_nrm < opts['stepAbsTol'],          # step / absolute
                        dx_nrm < opts['stepRelTol'] * x_nrm   # gradient / relative
                      ]
      stop = np.any(stop_criteria)
      # display
      if opts['verbose'] and (stop or itn == 1 or itn % opts['print_every'] == 0):
        itn_hdr += 1
        if itn_hdr == 1 or itn_hdr % opts['print_header_every'] == 0:
          print(header_str)
        if obj_x == np.inf:
          print(line_str_inf % (itn, 'inf', x_nrm, g_nrm, dx_nrm, t, ls_itns, x.imag.min(), x.imag.max()))
        else:
          print(line_str % (itn, obj_x, x_nrm, g_nrm, dx_nrm, t, ls_itns, x.imag.min(), x.imag.max()))
        
        if stop:
          print(f'\n')
          print(f'Termination criteria reached')
          for I, mssg in enumerate(self.termination):
            if stop_criteria[I]:
              print(' -> ' + mssg)
    # log
    self.x = x
    # return
    if xo is None:
      return x
    return x.reshape(xo.shape)

