#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thermo class for PRFS MR thermometry with optional TFI correction.

The class receives already-loaded complex MR signal data with shape
(x, y, z, echo, time). File loading should happen outside this class.
"""

import os
import numpy as np
import matlab.engine

from scipy.ndimage import binary_erosion, sobel
from matlabEngine import MatlabEngineArray2numpyArray
from unwrapping import UnwrappingWrapper
from helper import simulate_RDF_ppm

import funcLib

try:
    import cupy as xp
except ModuleNotFoundError:
    import numpy as xp


class Thermo:
    def __init__(
        self,
        signal,
        TE_s=np.array([4.604e-3, 18.415e-3], dtype=np.float32),
        voxelSize_mm=np.array([500 / 256, 500 / 256, 10], dtype=np.float32),
        fieldStrength_T=1.5,
        B0dir=np.array([0, 0, 1], dtype=np.float32),
        options=None,
        start_matlab=True,
    ):
        """
        Parameters
        ----------
        signal : np.ndarray
            Complex MR signal with shape (x, y, z, echo, time).
        TE_s : array-like
            Echo times in seconds. Must match the echo dimension of signal.
        voxelSize_mm : array-like
            Voxel size in mm, shape (3,).
        fieldStrength_T : float
            Main magnetic field strength in Tesla.
        B0dir : array-like
            Main field direction, usually [0, 0, 1].
        options : dict, optional
            Algorithm parameters overriding defaults in self.AlgoParams.
        start_matlab : bool
            If True, start MATLAB engine for polyfit3D.
        """
        self._signal = np.asarray(signal)

        self._TE_s = np.asarray(TE_s, dtype=np.float32)
        self._voxelSize_mm = np.asarray(voxelSize_mm, dtype=np.float32)
        self._fieldStrength_T = float(fieldStrength_T)
        self._B0dir = np.asarray(B0dir, dtype=np.float32)

        # Masks.
        self._tissueMask = None       # shape: (x, y, z, time-1)
        self._refTubeMask = None      # shape: (x, y, z)
        self._mask = None             # current 3D mask for TFI

        # Intermediate results.
        self._iFreq = None
        self._b0corr = None
        self._tempcorr = None
        self._RDF_ppm = None
        self._chimap_ppm = None
        self._localChimap_ppm = None
        self._currentSignal = None

        # TFI helper fields.
        self._paddingParams = {}
        self._preconditioner = None
        self._dataWeighting = None
        self._gradWeighting = None

        # Outputs.
        self.Tmaps = {}
        self.probe = {}

        # Physical constants.
        self._gamma = 2.68e8
        self._alpha = -0.00909e-6

        deltaTE = np.diff(self._TE_s)[0]
        self._temperatureFactor = (
            self._alpha * self._gamma * self._fieldStrength_T * deltaTE
        )
        self._correctionFactor = (self._TE_s[1] - self._TE_s[0]) / self._TE_s[0]

        self.AlgoParams = {
            "verbose": False,
            "referenceTimepoint": 1,
            "mask_threshold": 10,
            "nErosions": 0,
            "iEcho": False,
            "regularizationParameter": 1,
            "lp": 1.0,
            "max_cg_iter": 10,
            "max_iter": 20,
            "reltol_update": 0.01,
            "gradWeightingMethod": "sobel",
        }
        if options is not None:
            self.AlgoParams.update(options)

        self.matlab = None
        self._consistencyCheck()

        if start_matlab:
            self._setUpMatlab()

        self.set_tissueMask()

    # ------------------------------------------------------------
    # Setup / validation
    # ------------------------------------------------------------

    def _consistencyCheck(self):
        if self._signal.ndim != 5:
            raise ValueError("signal must have 5 dimensions: (x, y, z, echo, time)")

        if len(self._TE_s) < 2:
            raise ValueError("At least two echo times are required for PRFS thermometry")

        if self._signal.shape[3] != len(self._TE_s):
            raise ValueError("signal echo dimension must match len(TE_s)")

        if self._voxelSize_mm.shape != (3,):
            raise ValueError("voxelSize_mm must have shape (3,), e.g. [dx, dy, dz]")

        if self._B0dir.shape != (3,):
            raise ValueError("B0dir must have shape (3,), e.g. [0, 0, 1]")

        if self._signal.shape[4] < 2:
            raise ValueError("signal must contain at least two timepoints")

        if not np.iscomplexobj(self._signal):
            raise ValueError("signal should be complex-valued MR data")

        timeref = self.AlgoParams["referenceTimepoint"]
        if timeref < 0 or timeref >= self._signal.shape[4]:
            raise ValueError("referenceTimepoint is outside the available time dimension")

    def _setUpMatlab(self):
        eng = matlab.engine.start_matlab()
        path = os.path.dirname(os.path.abspath(__file__))
        eng.addpath(path)
        self.matlab = eng

    # ------------------------------------------------------------
    # Masking
    # ------------------------------------------------------------

    def set_tissueMask(self):
        """Create a dynamic tissue mask for each non-reference timepoint."""
        shape = self._signal.shape
        mask = np.zeros((*shape[0:3], shape[4] - 1), dtype=bool)

        threshold = self.AlgoParams["mask_threshold"]
        n_erosions = self.AlgoParams["nErosions"]

        for i in range(shape[4] - 1):
            signal_t = self._signal[:, :, :, :, i + 1]
            magnitude = np.mean(np.abs(signal_t), axis=-1)
            tmp_mask = magnitude > threshold / 100 * np.mean(magnitude)

            if n_erosions > 0:
                mask[..., i] = binary_erosion(tmp_mask, iterations=n_erosions)
            else:
                mask[..., i] = tmp_mask

        self._tissueMask = mask

    def set_refTubeMask(self, refTubeMask):
        refTubeMask = np.asarray(refTubeMask)
        if refTubeMask.shape != self._signal.shape[0:3]:
            raise ValueError("refTubeMask must have shape (x, y, z)")
        self._refTubeMask = refTubeMask.astype(bool)

    # ------------------------------------------------------------
    # Frequency / phase processing
    # ------------------------------------------------------------

    def calc_iFreq(self):
        """Calculate unwrapped phase/frequency-like maps relative to reference timepoint."""
        timeref = self.AlgoParams["referenceTimepoint"]
        signal = self._signal
        shape = signal.shape

        echoDiff_ref = signal[:, :, :, 1, timeref] * np.conj(signal[:, :, :, 0, timeref])
        
        cplx_timeref = signal[:, :, :, 0, timeref]

        iFreq = np.zeros((*shape[0:3], shape[4] - 1), dtype=np.float32)

        b0corr = None
        tempcorr = None
        Tmap = None
        if self._refTubeMask is not None:
            b0corr = np.zeros_like(iFreq)
            tempcorr = np.zeros_like(iFreq)
            Tmap = np.zeros_like(iFreq)

        if self.AlgoParams["iEcho"]:
            time_indices = [self.AlgoParams["iEcho"] + 1]
        else:
            time_indices = range(1, shape[4])

        for i in time_indices:
            if self.AlgoParams["verbose"]:
                print(i)

            if i == 1:
                iFreq[..., i - 1] = 0
            else:
                echoDiff = signal[:, :, :, 1, i] * np.conj(signal[:, :, :, 0, i])
            
                mask = self._tissueMask[..., i - 1]
                phase_diff_wrapped = np.angle(echoDiff * np.conj(echoDiff_ref))
                iFreq[..., i - 1] = self.unwrapping(phase_diff_wrapped, mask)

            if self._refTubeMask is not None:
                cplxdifftubes = signal[:, :, :, 0, i] * np.conj(cplx_timeref)

                b0corr[..., i - 1] = self.polyfit(
                    self._refTubeMask * np.angle(cplxdifftubes),
                    self._refTubeMask * np.abs(cplx_timeref),
                    1,
                )

                tempcorr[..., i - 1] = self.polyfit(
                    self._tissueMask[..., i - 1] * np.angle(cplxdifftubes),
                    self._tissueMask[..., i - 1] * np.abs(cplx_timeref),
                    0,
                )

                unwrapped = iFreq[..., i - 1] - self._correctionFactor * b0corr[..., i - 1]
                Tmap[..., i - 1] = unwrapped / self._temperatureFactor

        self._iFreq = iFreq
        if b0corr is not None:
            self._b0corr = b0corr
            self._tempcorr = tempcorr
            self.Tmaps["b0drift"] = Tmap

    def unwrapping(self, phase, mask, method="qgu"):
        return UnwrappingWrapper(phase, mask, method, self._voxelSize_mm)

    def polyfit(self, phase, reftubes, order):
        if self.matlab is None:
            raise RuntimeError(
                "MATLAB engine is not running. Initialize with start_matlab=True "
                "or call self._setUpMatlab()."
            )

        result = self.matlab.polyfit3D(
            matlab.double(phase.tolist()),
            matlab.double(reftubes.tolist()),
            order,
            nargout=1,
        )
        return MatlabEngineArray2numpyArray(result)

    # ------------------------------------------------------------
    # Temperature maps
    # ------------------------------------------------------------

    def set_temperatureUncorrected(self):
        if self._iFreq is None:
            self.calc_iFreq()
        self.Tmaps["uncorrected"] = self._iFreq / self._temperatureFactor

    def set_temperatureTFI(self):
        """Calculate TFI-corrected temperature maps."""
        if self._iFreq is None:
            self.calc_iFreq()

        tmap = np.zeros_like(self._iFreq)

        if self.AlgoParams["iEcho"]:
            indices = [self.AlgoParams["iEcho"]]
        else:
            indices = range(self._iFreq.shape[-1])

        for i in indices:
            self._RDF_ppm = self._iFreq[..., i].astype(np.float32)
            self._mask = self._tissueMask[..., i]
            self._currentSignal = self._signal[:, :, :, :, i + 1]

            self._setPaddingParams()
            self.calcPrecon()
            self.set_dataWeighting()
            self.set_gradWeighting(method=self.AlgoParams["gradWeightingMethod"])
            self.lTFI_lp()

            dB_tfi = simulate_RDF_ppm(
                self._chimap_ppm,
                self._voxelSize_mm,
                self._B0dir,
                zeropadding=True,
            )

            lftfi = self._trimPadArray(self._RDF_ppm) - dB_tfi
            lftfi = self._reverseTrimPadArray(lftfi)
            lftfi *= self._mask

            tmap[..., i] = lftfi / self._temperatureFactor

        self.Tmaps["tfi"] = tmap

    def reftubeCorrection(self):
        if "tfi" not in self.Tmaps:
            raise ValueError("Tmaps['tfi'] does not exist. Run set_temperatureTFI() first.")
        if self._refTubeMask is None:
            raise ValueError("No reference tube mask set. Use set_refTubeMask().")

        T_tfi = self.Tmaps["tfi"]
        for i in range(T_tfi.shape[-1]):
            mean = np.mean(T_tfi[..., i][self._refTubeMask])
            if self.AlgoParams["verbose"]:
                print(mean)
            T_tfi[..., i] -= mean

    def v0correction(self, keys=("lbv", "pdf")):
        if self._tempcorr is None:
            raise ValueError("self._tempcorr is not available. Run calc_iFreq() with a reference tube mask first.")

        for key in keys:
            if key not in self.Tmaps:
                continue
            tmap = self.Tmaps[key]
            for i in range(tmap.shape[-1]):
                val = np.mean(self._tempcorr[..., i])
                tmap[..., i] += val * self._correctionFactor / self._temperatureFactor

    # ------------------------------------------------------------
    # TFI / QSM helper methods
    # ------------------------------------------------------------

    def _setPaddingParams(self):
        paddingParams = self._paddingParams
        paddingParams["originalShape"] = self._signal.shape[0:3]

        array, slicing = funcLib.trim_zeros(self._mask)
        paddingParams["trimmedShape"] = array.shape
        paddingParams["slicingVals"] = slicing

        paddingParams["paddingVals"] = np.ceil(
            np.array(paddingParams["trimmedShape"]) / 4
        ).astype(np.int32)

        paddingParams["paddedShape"] = tuple(
            np.array(paddingParams["trimmedShape"]) + 2 * paddingParams["paddingVals"]
        )

        revSlicing, revPadding, revPadding_nopad = funcLib.calculateReverseCoefficients(
            paddingParams
        )
        paddingParams["reverseSlicingVals"] = revSlicing
        paddingParams["reversePaddingVals"] = revPadding
        paddingParams["reversePaddingValsNoPad"] = revPadding_nopad

    def _trimPadArray(self, arr):
        slicing = self._paddingParams["slicingVals"]
        padding = self._paddingParams["paddingVals"]

        if arr.shape == self._paddingParams["paddedShape"]:
            return arr

        if arr.ndim == 3:
            return funcLib.pad_array3d(arr[slicing], padding)

        if arr.ndim == 4:
            retArr = np.zeros((3, *self._paddingParams["paddedShape"]), dtype=arr.dtype)
            for i in range(3):
                retArr[i, ...] = funcLib.pad_array3d(arr[i, ...][slicing], padding)
            return retArr

        raise ValueError("arr must be 3D or 4D")

    def _reverseTrimPadArray(self, arr):
        paddingParams = self._paddingParams
        if arr.shape == paddingParams["originalShape"]:
            return arr

        return np.pad(
            arr[paddingParams["reverseSlicingVals"]],
            (
                paddingParams["reversePaddingVals"][0],
                paddingParams["reversePaddingVals"][1],
                paddingParams["reversePaddingVals"][2],
            ),
            "constant",
            constant_values=0,
        )

    def calcPrecon(self):
        DataParams = {
            "tissueMask": self._trimPadArray(self._mask),
            "voxelSize_mm": self._voxelSize_mm,
            "B0dir": self._B0dir,
            "RDF_ppm": self._trimPadArray(self._RDF_ppm),
        }
        self._preconditioner = funcLib.compute_tfipreconditioner(DataParams)

    def set_dataWeighting(self):
        MIP = self._getEchoMIP()
        weights = MIP / np.percentile(MIP, 95.0)
        weights[weights > 1] = 1.0
        self._dataWeighting = (weights * self._mask).astype(np.float32)

    def set_gradWeighting(self, method="gradient"):
        MIP = funcLib.move2gpu(self._getEchoMIP())

        if method == "sobel":
            edges = xp.zeros((3,) + MIP.shape).astype(xp.float32)
            for i in range(3):
                edges[i, ...] = xp.abs(sobel(MIP, axis=i))

            edges /= xp.percentile(edges, 98)
            edges[edges > 1] = 1.0
            edges = 1 - edges

        elif method == "gradient":
            edges = funcLib.compute_gradient_weights(MIP, self._voxelSize_mm, [1.0, 95.0, 99.0])

        else:
            raise ValueError(f"method {method} not supported")

        edges[edges < 0.1] = 0.1
        edges = funcLib.move2cpu(edges)

        for i in range(3):
            edges[i, ...] = edges[i, ...] * self._mask

        self._gradWeighting = funcLib.move2cpu(edges)

    def _getEchoMIP(self):
        if self._currentSignal is not None:
            return np.max(np.abs(self._currentSignal), axis=-1)
        return np.max(np.abs(self._signal), axis=(3, 4))

    def lTFI_lp(self):
        DataParams = {
            "localRDF_ppm": self._trimPadArray(self._RDF_ppm),
            "voxelSize_mm": self._voxelSize_mm,
            "B0dir": self._B0dir,
            "P": self._preconditioner if self._preconditioner is not None else 1,
        }

        Options = {
            "regularizationParameter": self.AlgoParams["regularizationParameter"],
            "dataWeighting": self._trimPadArray(self._dataWeighting),
            "gradWeighting": self._trimPadArray(self._gradWeighting),
            "max_cg_iter": self.AlgoParams.get("max_cg_iter", 10),
            "max_iter": self.AlgoParams.get("max_iter", 25),
            "reltol_update": self.AlgoParams.get("reltol_update", 0.01),
            "lp": self.AlgoParams.get("lp", 1),
            "verbose": self.AlgoParams.get("verbose", False),
        }

        chi = funcLib.TFI_linear_lp_dataconsistency(DataParams, Options)
        self._chimap_ppm = chi
        self._localChimap_ppm = chi * self._trimPadArray(self._mask)

    # ------------------------------------------------------------
    # Optional utilities
    # ------------------------------------------------------------

    def remove_zeropadding(self):
        raise NotImplementedError(
            "remove_zeropadding() requires depad_array2d and has not been ported yet."
        )


# ------------------------------------------------------------
# Optional standalone helpers
# ------------------------------------------------------------

def load_sorted_mat(filename, variable_name="images_b"):
    """Standalone loader. Use this outside Thermo, then pass signal to Thermo."""
    try:
        from bmrr_wrapper.Interfaces.ImDataParams.ImDataParamsBase import loadv73matlab
        return loadv73matlab(filename)[variable_name]
    except (OSError, ImportError):
        import scipy.io as sio
        return sio.loadmat(filename)[variable_name]


def masking3D(image, coeff=1):
    mag = np.abs(image)
    return mag > coeff * np.mean(mag)
