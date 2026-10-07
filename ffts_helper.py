#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 16:32:01 2026

@author: minglein
"""

import numpy as np
try:
    import cupy as cp
    from numpy.fft import fftn as cfftn
    from numpy.fft import ifftn as cifftn
    from numpy.fft import fftshift as cfftshift
    from numpy.fft import ifftshift as cifftshift

    from cupy.fft import fftn as gfftn
    from cupy.fft import ifftn as gifftn
    from cupy.fft import fftshift as gfftshift
    from cupy.fft import ifftshift as gifftshift

    def fftn(x):
        if isinstance(x, np.ndarray):
            y = cfftn(x)
        elif isinstance(x, cp.ndarray):
            y = gfftn(x)
        return y

    def ifftn(x):
        if isinstance(x, np.ndarray):
            y = cifftn(x)
        elif isinstance(x, cp.ndarray):
            y = gifftn(x)
        return y

    def fftshift(x):
        if isinstance(x, np.ndarray):
            y = cfftshift(x)
        elif isinstance(x, cp.ndarray):
            y = gfftshift(x)
        return y

    def ifftshift(x):
        if isinstance(x, np.ndarray):
            y = cifftshift(x)
        elif isinstance(x, cp.ndarray):
            y = gifftshift(x)
        return y

except ModuleNotFoundError:
    from numpy.fft import fftn, ifftn, fftshift, ifftshift
