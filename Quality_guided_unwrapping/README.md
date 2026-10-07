# 3D Quality-guided Unwraping     
# Copyright Véronique Fortier 2017 - MIT License 
# _______________________________________________

# Reference: V. Fortier and I. R. Levesque, Phase Processing for Quantitative Susceptibility Mapping of Regions With Large Susceptibility and Lack of Signal, Magn. Reson. Med., DOI 10.1002/mrm.26989, 2017.

# Technique adapted from:
# D. C. Ghiglia and M. D. Pritt, Two-Dimensional Phase Unwrapping: Theory, Algorithms and Software. New York: Wiley-Interscience, 1998.

# _______________________________________________
# To unwrap phase data, run qualityGuidedUnwrapping.m
# The head numerical susceptibility model used in the reference is included as a .mat file (susceptibilityHead_numericalModel.mat). Units are in ppm.

# -------------------------------------------------
# The whole head data from volunteer 1 and the pelvis data used in this publication are available at: https://osf.io/37gu5/