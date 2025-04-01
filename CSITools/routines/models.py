## ----------------------------------------------------------------------------
## Imports

import numpy as np
from numpy.typing import NDArray

# --- get_wf_model
#     gets chemical shifts and weights for a water/fat model
def get_wf_model(field : int | float,
                 mp    : bool         = False
                 ) -> tuple[list, list]:

  if mp:
    shifts = [
                [ 0.0 ],
                [ -3.80E-6 * 42.58E6 * field, 
                  -3.40E-6 * 42.58E6 * field, 
                  -2.60E-6 * 42.58E6 * field, 
                  -1.94E-6 * 42.58E6 * field, 
                  -0.39E-6 * 42.58E6 * field, 
                  +0.60E-6 * 42.58E6 * field ] 
            ]
    weights = [ 
                [ 1.0 ], 
                [ 0.087, 0.693, 0.128, 0.004, 0.039, 0.048 ] 
            ] 
  else:
    shifts = [ 
                [ 0.0 ], 
                [ -3.4E-6 * 42.58E6 * field ]
            ]
    weights = [ 
                [ 1.0 ], 
                [ 1.0 ] 
            ]
  return shifts, weights

def get_wfsilicone_model(field : int | float,
                 mp    : bool         = False
                 ) -> tuple[list, list]:

  if mp:
    shifts = [
                [ 0.0 ],
                [ -3.80E-6 * 42.58E6 * field, 
                  -3.40E-6 * 42.58E6 * field, 
                  -2.60E-6 * 42.58E6 * field, 
                  -1.94E-6 * 42.58E6 * field, 
                  -0.39E-6 * 42.58E6 * field, 
                  +0.60E-6 * 42.58E6 * field ],
                [ -4.90E-6 * 42.58E6 * field ]
            ]
    weights = [ 
                [ 1.000 ], 
                [ 0.087, 0.693, 0.128, 0.004, 0.039, 0.048 ],
                [ 1.000 ] 
            ] 
  else:
    shifts = [ 
                [ 0.0 ], 
                [ -3.4E-6 * 42.58E6 * field ],
                [ -4.9E-6 * 42.58E6 * field ]
            ]
    weights = [ 
                [ 1.0 ], 
                [ 1.0 ],
                [ 1.0 ]
            ]
  return shifts, weights

def get_wfsilicone_phantom_model(field : int | float,
                                 mp    : bool         = True
                                ) -> tuple[list, list]:

  temperature_shift_ppm = 0.14
  fat_freqs = [0.90, 1.30, 1.59, 2.03, 2.25, 2.77, 4.1, 4.3, 5.21, 5.31] 
  fat_freqs = [f - 4.7 - temperature_shift_ppm for f in fat_freqs] # in ppm, 4.7 ppm is the water peak

  # modelname='Berglund 10 peaks' for peanut oil's fat model
  cl = 17.97
  ndb = 3.48
  nmidb = 1.01

  ABC2J = [9, 6 * (cl - 4) - 8 * ndb + 2 * nmidb, 6,
                  4 * (ndb - nmidb), 6, 2 * nmidb, 2, 2, 1, 2 * ndb]
  ABC2J = np.array(ABC2J)
  # print(ABC2J)
  relamps = ABC2J[ABC2J != 0]
  relamps = relamps / np.sum(relamps)

  if mp:
      shifts = [
                  [ 0.0 ],
                  [ fat_freqs[0] * 1E-6 * 42.58E6 * field, 
                    fat_freqs[1] * 1E-6 * 42.58E6 * field, 
                    fat_freqs[2] * 1E-6 * 42.58E6 * field, 
                    fat_freqs[3] * 1E-6 * 42.58E6 * field, 
                    fat_freqs[4] * 1E-6 * 42.58E6 * field, 
                    fat_freqs[5] * 1E-6 * 42.58E6 * field, 
                    fat_freqs[6] * 1E-6 * 42.58E6 * field, 
                    fat_freqs[7] * 1E-6 * 42.58E6 * field,
                    fat_freqs[8] * 1E-6 * 42.58E6 * field,
                    fat_freqs[9] * 1E-6 * 42.58E6 * field ],
                  [ (-4.90E-6 - temperature_shift_ppm*1E-6) * 42.58E6 * field ]
              ]
      weights = [ 
                  [ 1.000 ], 
                  list(relamps),
                  [ 1.000 ] 
              ] 
  else:
      shifts = [ 
                  [ 0.0 ], 
                  [ (-3.4E-6 - temperature_shift_ppm*1E-6) * 42.58E6 * field ],
                  [ (-4.9E-6 - temperature_shift_ppm*1E-6) * 42.58E6 * field ]
              ]
      weights = [ 
                  [ 1.0 ], 
                  [ 1.0 ],
                  [ 1.0 ]
              ]
  return shifts, weights


# --- assemble_model_matrix
#     assembles the matrix associated to the echo times, and the chemical
#     shifts and weights
def assemble_model_matrix(te      : NDArray,      # (num_te,)   | float  
                          shifts  : list,         # num_sp      | list[float]
                          weights : list          # num_sp      | list[float]
                          ) -> NDArray:
  # number of echo times
  num_te = te.size
  # number of species
  num_sp = len(shifts)
  # validate input
  if len(weights) != num_sp:
      raise ValueError('The list of weights and chemical shifts must have the same length.')
  for _weights, _shifts in zip(weights, shifts):
      if len(_weights) != len(_shifts):
          raise ValueError('The number of chemical shifts and weights per species must be the same.')
  # assemble matrix
  M = np.zeros((num_te, num_sp), dtype=complex)
  for J, (_weights, _shifts) in enumerate(zip(weights, shifts)):
      M[:, J] = np.sum([ _w * np.exp(2 * np.pi * 1j * _f * te.flatten()) for _w, _f in zip(_weights, _shifts) ], axis=0)
  # return
  return M


