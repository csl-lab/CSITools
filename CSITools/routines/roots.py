## ----------------------------------------------------------------------------
## Imports

import numpy as np
from numpy.polynomial import Polynomial

import sympy

from numpy.typing import NDArray

from itertools import combinations

def find_roots_model(M : NDArray, te : NDArray, ceps : float = 1E-9, peps : float = 1E-4, feps : float = 1E-3, decimals : int = 6, efactor : int = 2, verbose : bool = False) -> NDArray:
  if te.size < 2 * M.shape[1]:
    raise ValueError('Matrix must have shape (num_echos, num_species) with num_echoes > 2 * num_species.')
  # number of species
  ns = M.shape[1]
  # number of echos
  ne = te.size
  # get rational approximation for echo times
  p_te = np.zeros((ne,), dtype=int)
  q_te = np.zeros((ne,), dtype=int)
  for I, _te in enumerate(te):
    (p, q) = sympy.nsimplify(_te / te[-1]).as_numer_denom()
    p_te[I] = p
    q_te[I] = q
  # lcm for denominator
  lcm_q_te = sympy.lcm(q_te)
  # integer exponents
  pe = (p_te * (lcm_q_te // q_te)).astype('int')
  if verbose:
    print('Summary')
    print('   lcm(q_k)      : ', lcm_q_te)
    print('   Exponents     : ', *pe)
    print('Sampling subsets')
  # polynomials
  pJM = []
  # model roots
  M_roots = []
  # iterate over subsets of rows
  tsubsets = list(combinations(set([ I for I in range(ne) ]), 2 * ns))
  for tsubset in tsubsets:
    # subset
    M_S = M[list(tsubset)]
    pe_S = pe[list(tsubset)]
    # reduced matrix
    M_S_red = M_S[ns:] @ np.linalg.inv(M_S[:ns])
    # * polynomial approximation using the fft
    # number of sampling points
    num_poly_coefs = efactor * pe_S.max()
    # sampling poins
    z = np.exp(-2 * np.pi * 1j * np.linspace(0, num_poly_coefs - 1, num_poly_coefs) / num_poly_coefs)
    # values
    pJz = np.array([ np.linalg.det(np.hstack([ np.diag(_z ** pe_S) @ M_S, M_S ])) for _z in z ], dtype=complex)
    # polynomial coefficients
    pJc = np.fft.ifft(pJz)
    # threshold coefficients and remove root at zero
    pJ_degree = np.where(np.abs(pJc) > ceps)[0].max()
    pJ_factor = np.where(np.abs(pJc) > ceps)[0].min()
    pJc = pJc[pJ_factor:pJ_degree + 1]
    # assemble polynomial
    pJ = Polynomial(pJc[::-1], domain=[-1, 1])
    # find roots
    pJ_rts = pJ.roots()
    # pair roots
    pJ_rts_paired = []
    for _z in pJ_rts:
      if np.abs(pJ(_z)) < peps and np.abs(pJ(1/_z)) < peps:
        pJ_rts_paired.append(_z)
    pJ_rts_paired = np.array(pJ_rts_paired)
    if verbose:
      print(f'   Sampling                : ', tsubset)
      print(f'   Exponents               : ', *pe_S)
      print(f'   Fiting degree           : ', num_poly_coefs)
      print(f'   Number of coefficients  : {pJc.size:d} ({np.sum(np.where(pJc > peps, 1.0, 0.0)):.0f} nonzero)')
      print(f'   Roots                   : {pJ_rts.size} (min |z| = {np.abs(pJ_rts).min():.3f}, max |z| = {np.abs(pJ_rts).max():.3f})')
      print(f'   Paired roots            : {pJ_rts_paired.size} (min |z| = {np.abs(pJ_rts_paired).min():.3f}, max |z| = {np.abs(pJ_rts_paired).max():.3f})')
    # append roots
    M_roots.append(pJ_rts_paired)
    # append polynomial
    pJM.append(pJ)
  # filter
  roots = []
  for _roots in M_roots:
    for _z in _roots:
      _pJz = np.array([ _pJ(_z) for _pJ in pJM ])
      if np.abs(_pJz).max() < peps:
        roots.append(_z) 
  roots = np.array(roots)
  # parameters
  pJ_eta = (np.arctan2(roots.imag, roots.real) - 1j * np.log(np.abs(roots))) / (2 * np.pi)
  # round
  pJ_eta = np.unique(np.round(pJ_eta, decimals=decimals))
  # filtering
  def sv_Delta(eta):
    W = np.diag(np.exp(2 * np.pi * 1j * te * eta))
    sv_Delta = np.linalg.svdvals(np.hstack([ W @ M, M ]))
    return sv_Delta.min(), np.sum(np.where(sv_Delta < feps, 1.0, 0.0))
  # filter by rank
  eta_offset_factor = float((lcm_q_te / te[-1]).evalf())
  pJ_eta_filt = []
  pJ_sv = np.zeros((pJ_eta.size, 2), dtype=float)
  for I, _pJ_eta in enumerate(pJ_eta):
    _svmin, _dimker = sv_Delta(_pJ_eta.real * eta_offset_factor)
    pJ_sv[I, 0] = _svmin
    pJ_sv[I, 1] = _dimker
    if _svmin < feps:
      pJ_eta_filt.append(_pJ_eta)
  pJ_eta_filt = np.array(pJ_eta_filt)
  pJ_eta_filt = np.unique(np.round(pJ_eta_filt, decimals=decimals))
  # return
  return pJ_eta, pJ_sv, pJ_eta_filt, float((lcm_q_te / te[-1]).evalf())
  