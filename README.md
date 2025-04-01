# CSITools: Tools for Chemical Shift Imaging

Implementation of tools for parameter reconstruction in Chemical Shift Imaging (CSI). This Magnetic Resonance Imaging (MRI) modality is also known as Chemical Shift Encoded MRI (CSE-MRI).

Version 1.0

Date: 31.03.2025

## References

The motivation behind the methods implemented and their analysis can be found in:

> C. Arrieta and C. A. Sing Long, "*Exact Local Recovery for Chemical Shift Imaging*." March 31, 2025.

The code reproducing the results in these references can be found on ``experiments``.

### Citation

You can use the following code to cite our work.

```bibtex
@article{arrieta_exact_2025, 
  title   = {Exact Local Recovery for Chemical Shift Imaging},  
  author  = {Arrieta, Cristóbal and Sing Long, Carlos A.}, 
  date    = {2025-03-31}
}
```

## Repository

### Dependencies

The dependencies for the Python implementation are:
* ``numpy``
* ``scipy``
* ``cvxpy``
* ``h5py``
* ``matplotlib``

The repository uses a publicly available data for the *in vitro* phantom introduced in the paper:

> J. K. Stelter, C. Boehm, S. Ruschke, K. Weiss, M. N. Diefenbach, M. Wu, T. Borde, G. P. Schmidt, M. R.
Makowski, E. M. Fallenberg, and D. C. Karampinos, "Hierarchical multi-resolution graph-cuts for water-fat-
silicone separation in breast MRI," IEEE Transactions on Medical Imaging, vol. 41, no. 11, pp. 3253–3265, Nov. 2022. DOI: [10.1109/TMI.2022.3180302](https://doi.org/10.1109/TMI.2022.3180302).

The data for the *in vitro* phantom is not provided in this repository but it can be found [here](https://syncandshare.lrz.de/getlink/fi2yT7Vp761X2EW2XbY41KnM/). **We kindly request you to cite the above paper if you use this data**.

### Structure

``CSITools\``
* ``objects\``
    * ``csimap.py``: objects defining the Chemical Shift Imaging Map (CSIMap)
    * ``prox.py``: objects defining prox-capable regularizers
    * ``smooth.py``: objects defining smooth functions
    * ``solver.py``: objects defining generic solvers
* ``routines\``
    * ``csimap.py``: routines to evaluate the Chemical Shift Imaging Map (CSIMap)
    * ``gradients.py``: matrices to compute partial derivatives
    * ``loader.py``: utilities to load data
    * ``models.py``: functions to retrieve models for the chemical species
    * ``residual.py``: routines to evaluate the residuals for the CSIMap
    * ``roots.py``: utilities to compute the structure of the solution set
    * ``signal.py``: utilities to simulate signals and retrieve the concentrations

``experiments\``
  * ``exactRecovery\``: experiments in "*Exact Local Recovery for Chemical Shift Imaging*"
    * ``FIG\``: folder in which the figures will be saved
    * ``DAT\``: folder with precomputed data to generate the figures
    * ``01_Solution_Set.ipynb``: structure of the solution set for a water and fat, and water, fat and silicone model
    * ``02_Local_recovery.ipynb``: finds the radius around which the residual has positive curvature around the true parameter for a water, fat and silicone phantom
    * ``03_In_silico_phantom.ipynb``: performs recovery from data generated from an *in silico* water, fat and silicone phantom
    * ``04_In_vitro_phantom.ipynb``: performs recovery from data generated from an *in vitro* water, fat and silicone phantom
 
