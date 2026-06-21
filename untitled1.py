#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 10:11:19 2026

@author: minglein
"""

import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, '/Users/minglein/Documents/Code')
sys.path.insert(0, '/Users/minglein/Documents/Code/bmrrpython')
sys.path.insert(0, '/Users/minglein/Documents/Code/bmrrpython/water-fat-imaging/css')
sys.path.insert(0, '/Users/minglein/Documents/Code/2020-thermometry-master/code')

from bmrr_shared_helper.general_helper import files_in_path, load_nii_array, scale_array2interval
#import pycss
from thermoclass import thermo
import bmrr_wrapper.Visualization.itkutils as iu

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import binary_dilation


#base_dir = Path("/Users/minglein/Documents/DATA/RFHT/Pelvis_patients")
base_dir = Path("/Users/minglein/Documents/DATA/RFHT/forMingming3")


def plot_qc(img5d, mask3d, patient_id, outdir=None, z=None, echo=0, time=0):
    vol = img5d[:, :, :, echo, time]

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
    plt.title(f"{patient_id}\nimage + mask overlay, z={z}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask_slice, cmap="gray", origin="lower")
    plt.title("reference mask")
    plt.axis("off")

    plt.tight_layout()

    if outdir is not None:
        save_path = Path(outdir) / f"{patient_id}_QC_mask_overlay.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close()


def save_individual_tempcorr_plots(wlobj, patient_dir, dt=0.1511):
    """
    Saves one curve per z-slice using mean tempcorr over x/y.
    """
    outdir = Path(patient_dir)
    patient_id = outdir.name

    n_time = wlobj.tempcorr.shape[-1]
    time_axis = np.arange(n_time) * dt

    n_slices = wlobj.tempcorr.shape[2]

    for z in range(n_slices):
        y = np.array([
            np.mean(wlobj.tempcorr[:, :, z, t])
            for t in range(n_time)
        ])

        plt.figure()
        plt.plot(time_axis, y)
        plt.xlabel("Time (s)")
        plt.ylabel("Mean tempcorr")
        plt.title(f"{patient_id} - slice {z}")
        plt.tight_layout()

        save_path = outdir / f"{patient_id}_tempcorr_slice_{z}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


def process_patient(patient_dir):
    patient_id = patient_dir.name
    print(f"\nProcessing {patient_id}")

#    mat_path = patient_dir / f"{patient_id}.mat"
    mat_path = patient_dir / "images_b_v7.mat"
    mask_path = patient_dir / "reftubes.nii.gz"
    save_path = patient_dir / "results_iter20.mat"

    if not mat_path.exists():
        print(f"  [SKIP] Missing {mat_path.name}")
        return

    if not mask_path.exists():
        print(f"  [SKIP] Missing {mask_path.name}")
        return

    wlobj = thermo(str(mat_path))

    refmask = load_nii_array(str(mask_path)).astype(bool)
    print(f"  raw refmask shape: {refmask.shape}")

    # We found that this transpose was needed for the patient-folder data
    # when raw mask shape was (25, 256, 256) and image shape was (256, 256, 25, ...)
    if refmask.shape != wlobj.data.shape[:3]:
        refmask_t = np.transpose(refmask, (2, 1, 0))
        if refmask_t.shape == wlobj.data.shape[:3]:
            refmask = refmask_t
            print("  applied transpose: (2, 1, 0)")

    print(f"  data shape:    {wlobj.data.shape}")
    print(f"  refmask shape: {refmask.shape}")

    if refmask.shape != wlobj.data.shape[:3]:
        raise ValueError(
            f"Mask shape {refmask.shape} does not match image shape {wlobj.data.shape[:3]}"
        )

    plot_qc(wlobj.data, refmask, patient_id, outdir=patient_dir)

    wlobj.refTubeMask = refmask

    options = {
        'mask_threshold': 10,
        'nErosions': 1
    }

    wlobj._setOptions(options)
    wlobj._setMask()
    wlobj._calciFreq()

    for i in range(wlobj.tempcorr.shape[-1]):
        print(f"  mean tempcorr[{i}]: {np.mean(wlobj.tempcorr[..., i])}")

    wlobj.set_temperatureUncorrected()
    # wlobj.set_temperaturePDF()
    # wlobj.set_temperatureLBV()
    wlobj.set_temperatureTFI()

    wlobj.reftubeCorrection()

    save_individual_tempcorr_plots(wlobj, patient_dir, dt=0.1511)

    if hasattr(wlobj, "matlab"):
        delattr(wlobj, "matlab")

    sio.savemat(str(save_path), wlobj.Tmaps)
    print(f"  saved: {save_path}")


for patient_dir in sorted(base_dir.iterdir()):
    if not patient_dir.is_dir():
        continue

    try:
        process_patient(patient_dir)
    except Exception as e:
        print(f"  [ERROR] {patient_dir.name}: {e}")
        traceback.print_exc()