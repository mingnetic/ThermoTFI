#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal numerical helper functions for thermo.py.

This file intentionally contains only the functions needed by the Thermo class:
CPU/GPU helpers, FFT wrappers, padding/trimming utilities, dipole kernel,
finite differences, conjugate-gradient solver, TFI preconditioner, and the
LP data-consistency TFI solver.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import curve_fit

EPS = 1e-6
GAMMA_BAR_MHZ_T = 42.577478518

try:
    import cupy as xp
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
            return cfftn(x)
        if isinstance(x, cp.ndarray):
            return gfftn(x)
        raise TypeError("Input must be a numpy or cupy array")

    def ifftn(x):
        if isinstance(x, np.ndarray):
            return cifftn(x)
        if isinstance(x, cp.ndarray):
            return gifftn(x)
        raise TypeError("Input must be a numpy or cupy array")

    def fftshift(x):
        if isinstance(x, np.ndarray):
            return cfftshift(x)
        if isinstance(x, cp.ndarray):
            return gfftshift(x)
        raise TypeError("Input must be a numpy or cupy array")

    def ifftshift(x):
        if isinstance(x, np.ndarray):
            return cifftshift(x)
        if isinstance(x, cp.ndarray):
            return gifftshift(x)
        raise TypeError("Input must be a numpy or cupy array")

    def move2cpu(x, xp_module=cp):
        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
        return x

    def move2gpu(x, xp_module=cp):
        if isinstance(x, np.ndarray):
            return cp.asarray(x)
        return x

except ModuleNotFoundError:
    import numpy as xp
    cp = None
    from numpy.fft import fftn, ifftn, fftshift, ifftshift

    def move2cpu(x, xp_module=np):
        return x

    def move2gpu(x, xp_module=np):
        return x


def _array_module(x):
    if cp is not None and isinstance(x, cp.ndarray):
        return cp
    if isinstance(x, np.ndarray):
        return np
    raise TypeError("Input must be a numpy or cupy array")


def conjugate_gradient(A, b, x=None, precond=None, max_iter=512, reltol=1e-2, verbose=False):
    """Conjugate-gradient solver for Ax=b, where A is a function."""
    xp_local = _array_module(b)

    if verbose:
        print("Starting conjugate gradient...")

    if x is None:
        x = xp_local.zeros_like(b)

    if precond is None:
        r = b - A(x)
        d = r.copy()
        rsnew = xp_local.sum(r.conj() * r).real
        rs0 = rsnew

        if verbose:
            print(f"initial residual: {rsnew}")

        ii = 0
        while ii < max_iter and rsnew > (reltol**2 * rs0):
            ii += 1
            Ad = A(d)
            alpha = rsnew / xp_local.sum(d.conj() * Ad)
            x = x + alpha * d

            if ii % 50 == 0:
                r = b - A(x)
                d = r.copy()
            else:
                r = r - alpha * Ad

            rsold = rsnew
            rsnew = xp_local.sum(r.conj() * r).real
            d = r + rsnew / rsold * d

            if verbose:
                print(f"{ii}, residual: {rsnew}")

    else:
        r = b - A(x)
        r_invM_r_old = (r.conj() * precond(r)).sum()
        res0 = xp_local.linalg.norm(r)
        p = precond(r)

        if verbose:
            print(f"initial residual: {res0}")

        ii = 0
        while ii <= max_iter:
            ii += 1
            Ap = A(p)
            alpha = r_invM_r_old / (p.conj() * Ap).real.sum()
            x += alpha * p

            if ii % 50 == 0:
                r = b - A(x)
                p = precond(r)
            else:
                r -= alpha * Ap

            res = xp_local.linalg.norm(r)
            if res < reltol * res0:
                break

            r_invM_r_new = (precond(r).conj() * r).sum()
            beta = r_invM_r_new / r_invM_r_old
            r_invM_r_old = r_invM_r_new
            p = precond(r) + beta * p

            if verbose:
                print(f"step {ii}: residual = {res}")

    return x


def fdiff(x, delta=1.0, axis=0):
    xp_local = _array_module(x)
    return (xp_local.roll(x, 1, axis=axis) - x) / delta


def fdiff_hc(x, delta=1.0, axis=0):
    xp_local = _array_module(x)
    return (xp_local.roll(x, -1, axis=axis) - x) / delta


def get_dipoleKernel_kspace(matrixSize, voxelSize_mm, B0dir, DCoffset=0):
    matrixSize = list(matrixSize)
    voxelSize_mm = xp.asarray(voxelSize_mm)
    B0dir = xp.asarray(B0dir)

    i = xp.linspace(-matrixSize[0] // 2, matrixSize[0] // 2 - 1, matrixSize[0])
    j = xp.linspace(-matrixSize[1] // 2, matrixSize[1] // 2 - 1, matrixSize[1])
    k = xp.linspace(-matrixSize[2] // 2, matrixSize[2] // 2 - 1, matrixSize[2])
    J, I, K = xp.meshgrid(j, i, k)

    dk = [1 / (voxelSize_mm[n] * matrixSize[n]) for n in range(3)]
    Ki = dk[0] * I
    Kj = dk[1] * J
    Kk = dk[2] * K

    Kz = B0dir[0] * Ki + B0dir[1] * Kj + B0dir[2] * Kk
    K2 = Ki**2 + Kj**2 + Kk**2

    center = K2 == 0
    K2[center] = xp.inf

    D = 1 / 3 - Kz**2 / K2
    D[center] = DCoffset
    return fftshift(D).astype(xp.float32)


def pad_array3d(arr, padsize, xp_module=xp):
    """Symmetrically zero-pad a 3D array."""
    arr = move2gpu(arr, xp_module)
    arr_big = xp_module.pad(
        arr,
        ((padsize[0], padsize[0]), (padsize[1], padsize[1]), (padsize[2], padsize[2])),
        "constant",
        constant_values=0,
    )
    return move2cpu(arr_big, xp_module)


def trim_zeros(arr, margin=0):
    """Trim leading/trailing empty planes from an N-D array."""
    s = []
    for dim in range(arr.ndim):
        start = 0
        end = -1
        slice_ = [slice(None)] * arr.ndim

        go = True
        while go:
            slice_[dim] = start
            go = not np.any(arr[tuple(slice_)])
            start += 1
            if start >= arr.shape[dim]:
                raise ValueError("Cannot trim an array that is empty in at least one dimension")
        start = max(start - 1 - margin, 0)

        go = True
        while go:
            slice_[dim] = end
            go = not np.any(arr[tuple(slice_)])
            end -= 1
        end = arr.shape[dim] + min(-1, end + 1 + margin) + 1

        s.append(slice(start, end))

    return arr[tuple(s)], tuple(s)


def calculateReverseCoefficients(paddingParams):
    slicing = []
    padding = []
    padding_nopad = []

    for i in range(3):
        x1_pad = 0
        x2_pad = 0
        x1 = paddingParams["paddingVals"][i] - paddingParams["slicingVals"][i].start
        x2 = (
            paddingParams["paddedShape"][i]
            - paddingParams["paddingVals"][i]
            + paddingParams["originalShape"][i]
            - paddingParams["slicingVals"][i].stop
        )

        if x1 < 0:
            x1_pad = abs(x1)
            x1 = 0
        if x2 > paddingParams["paddedShape"][i]:
            x2_pad = x2 - paddingParams["paddedShape"][i]

        x1_nopad = paddingParams["slicingVals"][i].start
        x2_nopad = paddingParams["originalShape"][i] - paddingParams["slicingVals"][i].stop

        slicing.append(slice(x1, x2))
        padding.append((x1_pad, x2_pad))
        padding_nopad.append((x1_nopad, x2_nopad))

    return tuple(slicing), tuple(padding), tuple(padding_nopad)


def BFRPDF(fieldmap_ppm, B0dir, voxelSize_mm, mask_tissue, max_iter=20):
    """Projection onto dipole fields background-field removal."""
    fieldmap_ppm = move2gpu(fieldmap_ppm).astype(xp.float32)
    mask_tissue = move2gpu(mask_tissue).astype(bool)

    D = get_dipoleKernel_kspace(fieldmap_ppm.shape, voxelSize_mm, B0dir)
    mask_bfr = xp.invert(mask_tissue)

    b = mask_bfr * xp.real(ifftn(D * fftn(mask_tissue * fieldmap_ppm))).astype(xp.float32)

    def A(x):
        fB = xp.real(ifftn(D * fftn(mask_bfr * x)))
        return mask_bfr * xp.real(ifftn(D * fftn(mask_tissue * fB)))

    chi_background_ppm = conjugate_gradient(A, b, max_iter=max_iter).astype(xp.float32)
    background_fieldmap_ppm = xp.real(ifftn(D * fftn(chi_background_ppm))).astype(xp.float32)
    local_fieldmap_ppm = (fieldmap_ppm - background_fieldmap_ppm).astype(xp.float32) * mask_tissue

    return (
        move2cpu(chi_background_ppm),
        move2cpu(background_fieldmap_ppm),
        move2cpu(local_fieldmap_ppm),
    )


def compute_tfipreconditioner(DataParams):
    """Compute TFI preconditioner from tissue mask and RDF field."""
    mask_tissue = DataParams["tissueMask"].astype(bool)
    voxelSize_mm = DataParams["voxelSize_mm"]
    B0dir = DataParams["B0dir"]
    field_ppm = DataParams["RDF_ppm"].astype(np.float32)

    mask_bgr = np.invert(mask_tissue)
    distancemap = distance_transform_edt(mask_bgr, voxelSize_mm)

    xbgr, _, _ = BFRPDF(field_ppm, B0dir, voxelSize_mm, mask_tissue, max_iter=10)
    xbgr = move2cpu(xbgr)

    dmin, dmax = 0.0, 100.0
    numbins = 100
    bins = np.linspace(dmin, dmax, num=numbins + 1)
    distances = []
    chibgrs = []

    for i in range(numbins):
        sel = np.logical_and(distancemap > bins[i], distancemap <= bins[i + 1])
        if sel.sum() > 0:
            distances.append(distancemap[sel].mean())
            chibgrs.append(np.median(np.abs(xbgr[sel])))

    def cubic_decay(r, r0, s0):
        return s0 / (1 + r / r0) ** 3

    if len(distances) >= 2:
        popt, _ = curve_fit(cubic_decay, distances, chibgrs, (45.0, 0.7))
    else:
        popt = (45.0, 0.7)

    ptfi = np.zeros_like(field_ppm, dtype=np.float32)
    ptfi_max = 30
    ptfi[mask_bgr] = ptfi_max / popt[1] * cubic_decay(distancemap[mask_bgr], *popt)
    ptfi[mask_tissue] = 1

    return ptfi.astype(np.float32)


def compute_gradient_weights(mag, voxelSize_mm, percentiles=(1.0, 95.0, 99.0)):
    """Compute 3D gradient weights from a magnitude image."""
    mag = move2gpu(mag)
    mag_gradients = xp.array([fdiff(mag, voxelSize_mm[i], axis=i) for i in range(mag.ndim)])

    valid = mag > 0.1
    for mg in mag_gradients:
        mgabs = xp.abs(mg)
        mgmin, mgthresh, mgmax = xp.percentile(mgabs[valid], percentiles)
        mgabs = (mgabs - mgmin) / (mgmax - mgmin + EPS)
        mgthresh = (mgthresh - mgmin) / (mgmax - mgmin + EPS)
        mg[:] = xp.maximum(mgthresh - mgabs, 0.0) / (mgthresh + EPS)

    return mag_gradients


def TFI_linear_lp_dataconsistency(DataParams, Options):
    """LP data-consistency TFI solver.

    Expected DataParams keys:
        localRDF_ppm, voxelSize_mm, B0dir, P

    Expected Options keys:
        regularizationParameter, dataWeighting, gradWeighting,
        max_cg_iter, max_iter, reltol_update, lp
    """
    psi = move2gpu(DataParams["localRDF_ppm"]).astype(xp.float32)

    voxelSize_mm = DataParams["voxelSize_mm"]
    B0dir = DataParams["B0dir"]

    max_cg_iter = Options.get("max_cg_iter", 10)
    max_iter = Options.get("max_iter", 25)
    reltol_update = Options.get("reltol_update", 0.01)
    lamda = Options["regularizationParameter"]
    lp = Options.get("lp", 1)

    P = DataParams.get("P", 1)
    if not isinstance(P, int):
        P = move2gpu(P)

    if "initChi_ppm" in DataParams:
        x = move2gpu(DataParams["initChi_ppm"])
        y = x / P
    else:
        y = xp.zeros_like(psi)

    W = move2gpu(Options["dataWeighting"])
    M = move2gpu(Options["gradWeighting"])
    W = xp.asarray(W)

    D = get_dipoleKernel_kspace(psi.shape, voxelSize_mm, B0dir)

    for t_outer in range(max_iter):
        modMGPy = xp.zeros_like(y)
        MGpy = xp.zeros((3, *y.shape), dtype=xp.float32)

        for i in range(3):
            MGpy[i] = M[i, ...] * fdiff(P * y, axis=i, delta=voxelSize_mm[i])
            modMGPy += MGpy[i] ** 2
        modMGPy = 1 / xp.sqrt(modMGPy + P**2 * EPS)

        DPy = ifftn(D * fftn(P * y)).real
        z = xp.sqrt((W * DPy) ** 2 + P**2 * EPS)
        modWDPy = 1 / z

        b = (
            lp / 2
            * P
            * ifftn(D * fftn(W * z ** (lp - 1) * modWDPy * W * (psi - DPy))).real
        )

        for i in range(3):
            b -= lamda * P * fdiff_hc(
                M[i, ...] * modMGPy * MGpy[i],
                axis=i,
                delta=voxelSize_mm[i],
            )

        def A(dy):
            lhs = (
                lp / 2
                * P
                * ifftn(
                    D
                    * fftn(
                        W
                        * z ** (lp - 1)
                        * modWDPy
                        * W
                        * ifftn(D * fftn(P * dy)).real
                    )
                ).real
            )
            for i in range(3):
                MGPdyi = M[i, ...] * fdiff(P * dy, axis=i, delta=voxelSize_mm[i])
                lhs += lamda * P * fdiff_hc(
                    M[i, ...] * modMGPy * MGPdyi,
                    axis=i,
                    delta=voxelSize_mm[i],
                )
            return lhs

        dy = conjugate_gradient(A, b, max_iter=max_cg_iter)
        y += dy

        ynorm = xp.linalg.norm(y)
        dynorm = xp.linalg.norm(dy)

        if Options.get("verbose", False):
            print(f"Iter: {t_outer}, update: {dynorm / ynorm}")

        if dynorm / ynorm < reltol_update:
            break

    return move2cpu(P * y)
