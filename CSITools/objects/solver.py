## ----------------------------------------------------------------------------
## Imports

import numpy as np

from abc import abstractmethod

np.seterr(over = 'raise', under = 'raise')

# --- CSIToolsSolver
#     Main class for solvers
class CSIToolsSolver(object):
  def __init__(self):
    pass
  
  @abstractmethod
  def solve(self, xo):
    pass