#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal one-patient test script for the refactored Thermo class.

Expected folder contents
------------------------
patient_dir/
    images_b_v7.mat
    reftubes.nii.gz

The Thermo class expects an already-loaded complex signal array with shape:
    (x, y, z, echo, time)
"""

import sys
import traceback
from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# Adjust these paths to your local setup
# -------------------------------------------------------------------------
CODE_DIR = Path("/Users/minglein/Documents/Code")
#BMRR_DIR = Path("/Users/minglein/Documents/Code/bmrrpython")

sys.path.insert(0, str(CODE_DIR))
#sys.path.insert(0, str(BMRR_DIR))

# Put thermo.py and funcLib.py either in the same folder as this script,
# or add their folder here:
# sys.path.insert(0, "/path/to/your/new/thermo_toolbox")

from general_helper import load_nii_array
from thermo import Thermo, load_sorted_mat


# -------------------------------------------------------------------------
# User settings
# -------------------------------------------------------------------------
BASE_DIR = Path("/Users/minglein/Documents/DATA/RFHT/testPhantomLMU")
PATIENT_ID = None  # e.g. "Patient001"; if None, the first patient folder is used

MAT_FILENAME = "images_b_v7.mat"
REFMASK_FILENAME = "reftubes.nii.gz"
RESULTS_FILENAME = "results_thermo_refactored.mat"

TE_S = np.array([4.604e-3, 18.415e-3], dtype=np.float32)
VOXEL_SIZE_MM = np.array([500 / 256, 500 / 256, 10], dtype=np.float32)
FIELD_STRENGTH_T = 1.5

THERMO_OPTIONS = {
    "mask_threshold": 10,
    "nErosions": 1,
    "referenceTimepoint": 1,
    "regularizationParameter": 1,
    "lp": 1.0,
    "max_iter": 20,
    "max_cg_iter": 10,
    "reltol_update": 0.01,
    "gradWeightingMethod": "sobel",
    "verbose": True,
}


# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------
def find_one_patient(base_dir: Path, patient_id: str | None = None) -> Path:
    if patient_id is not None:
        patient_dir = base_dir / patient_id
        if not patient_dir.is_dir():
            raise FileNotFoundError(f"Patient folder does not exist: {patient_dir}")
        return patient_dir

    patient_dirs = sorted(p for p in base_dir.iterdir() if p.is_dir())
    if not patient_dirs:
        raise FileNotFoundError(f"No patient folders found in {base_dir}")
    return patient_dirs[0]


def load_reference_mask(mask_path: Path, image_shape_3d: tuple[int, int, int]) -> np.ndarray:
    refmask = load_nii_array(str(mask_path)).astype(bool)
    print(f"  raw refmask shape: {refmask.shape}")

    # For your current patient-folder data, masks may be stored as (z, y, x)
    # while images are (x, y, z).
    if refmask.shape != image_shape_3d:
        refmask_t = np.transpose(refmask, (2, 1, 0))
        if refmask_t.shape == image_shape_3d:
            refmask = refmask_t
            print("  applied transpose: (2, 1, 0)")

    if refmask.shape != image_shape_3d:
        raise ValueError(
            f"Mask shape {refmask.shape} does not match image shape {image_shape_3d}"
        )

    return refmask


def plot_qc(signal5d, mask3d, patient_id, outdir=None, z=None, echo=0, time=0):
    vol = signal5d[:, :, :, echo, time]
    if np.iscomplexobj(vol):
        vol = np.abs(vol)

    if z is None:
        z = vol.shape[2] // 2

    img_slice = vol[:, :, z]
    mask_slice = mask3d[:, :, z]

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(img_slice, cmap="gray", origin="lower")
    plt.imshow(mask_slice, cmap="Reds", alpha=0.3, origin="lower")
    plt.title(f"{patient_id}\nimage + ref mask, z={z}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask_slice, cmap="gray", origin="lower")
    plt.title("reference mask")
    plt.axis("off")

    plt.tight_layout()

    if outdir is not None:
        save_path = Path(outdir) / f"{patient_id}_QC_mask_overlay_refactored.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  saved QC plot: {save_path}")

    plt.show()
    plt.close()


def save_individual_tempcorr_plots(thermo_obj: Thermo, patient_dir: Path, dt=0.1511):
    tempcorr = thermo_obj._tempcorr
    if tempcorr is None:
        print("  no tempcorr available; skipping tempcorr plots")
        return

    patient_id = patient_dir.name
    n_time = tempcorr.shape[-1]
    time_axis = np.arange(n_time) * dt
    n_slices = tempcorr.shape[2]

    for z in range(n_slices):
        y = np.array([np.mean(tempcorr[:, :, z, t]) for t in range(n_time)])

        plt.figure()
        plt.plot(time_axis, y)
        plt.xlabel("Time (s)")
        plt.ylabel("Mean tempcorr")
        plt.title(f"{patient_id} - slice {z}")
        plt.tight_layout()

        save_path = patient_dir / f"{patient_id}_tempcorr_slice_{z}_refactored.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    print(f"  saved tempcorr plots for {n_slices} slices")


# -------------------------------------------------------------------------
# Main processing
# -------------------------------------------------------------------------
def process_one_patient(patient_dir: Path):
    patient_id = patient_dir.name
    print(f"\nProcessing {patient_id}")

    mat_path = patient_dir / MAT_FILENAME
    mask_path = patient_dir / REFMASK_FILENAME
    save_path = patient_dir / RESULTS_FILENAME

    if not mat_path.exists():
        raise FileNotFoundError(f"Missing MAT file: {mat_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing reference mask: {mask_path}")

    signal = load_sorted_mat(str(mat_path), variable_name="images_b")
    print(f"  signal shape: {signal.shape}")

    refmask = load_reference_mask(mask_path, signal.shape[:3])
    print(f"  refmask shape: {refmask.shape}")

    plot_qc(signal, refmask, patient_id, outdir=patient_dir)

    thermo = Thermo(
        signal=signal,
        TE_s=TE_S,
        voxelSize_mm=VOXEL_SIZE_MM,
        fieldStrength_T=FIELD_STRENGTH_T,
        options=THERMO_OPTIONS,
        start_matlab=True,
    )

    thermo.set_refTubeMask(refmask)
    thermo.set_tissueMask()
    thermo.calc_iFreq()

    if thermo._tempcorr is not None:
        for i in range(thermo._tempcorr.shape[-1]):
            print(f"  mean tempcorr[{i}]: {np.mean(thermo._tempcorr[..., i])}")

    thermo.set_temperatureUncorrected()
    thermo.set_temperatureTFI()
    thermo.reftubeCorrection()

    save_individual_tempcorr_plots(thermo, patient_dir, dt=0.1511)

    sio.savemat(
        str(save_path),
        {
            "Tmaps": thermo.Tmaps,
            "iFreq": thermo._iFreq,
            "tempcorr": thermo._tempcorr,
            "b0corr": thermo._b0corr,
            "tissueMask": thermo._tissueMask,
            "refTubeMask": thermo._refTubeMask,
        },
    )
    print(f"  saved: {save_path}")

    # Optional: stop MATLAB engine cleanly.
    if thermo.matlab is not None:
        try:
            thermo.matlab.quit()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        patient_dir = find_one_patient(BASE_DIR, PATIENT_ID)
        process_one_patient(patient_dir)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        traceback.print_exc()
        raise
