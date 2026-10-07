#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 16:27:37 2026

@author: minglein
"""

import os
import h5py as h5
import hdf5storage
import numpy as np
try:
    import cupy as xp
except ModuleNotFoundError:
    import numpy as xp

from ffts_helper import fftn, ifftn, fftshift, ifftshift
#from bmrr_shared_helper.padding import depad_array3d, pad_array3d, \
#    trim_zeros, revert_trim_zeros
#from bmrr_shared_helper.solver import conjugate_gradient
from bmrr_shared_helper.general_helper import move2cpu, move2gpu

#from scipy.ndimage.morphology import distance_transform_edt
#from scipy.ndimage import label, measurements, morphology
#from scipy.optimize import curve_fit

#import bmrr_wrapper.Visualization.itkutils as iu
#from bmrr_wrapper.Interfaces.ImDataParams.ImDataParamsMasks import remove_neighborlessVoxelsFromMask

def depad_array3d(arr, padsize):
    ''':param arr: numpy array :returns: depadded array symmetrically with size
    given in padsize (per dimension)

    Christof Boehm,
    christof.boehm@tum.de

    '''
    kernelSize = arr.shape
    arrSmall = arr[padsize[0]:(kernelSize[0]-padsize[0]), \
                   padsize[1]:(kernelSize[1]-padsize[1]), \
                   padsize[2]:(kernelSize[2]-padsize[2])]
    return arrSmall

def get_dipoleKernel_kspace(matrixSize, voxelSize_mm, B0dir, DCoffset=0):

    matrixSize = list(matrixSize)
    voxelSize_mm = list(voxelSize_mm)

    i = xp.linspace(-matrixSize[0]//2, matrixSize[0]//2 - 1, matrixSize[0])
    j = xp.linspace(-matrixSize[1]//2, matrixSize[1]//2 - 1, matrixSize[1])
    k = xp.linspace(-matrixSize[2]//2, matrixSize[2]//2 - 1, matrixSize[2])
    J, I, K = xp.meshgrid(j, i, k)


    dk = [1/(a*b) for a,b in zip(voxelSize_mm, matrixSize)]
    Ki = dk[0].item() * I
    Kj = dk[1].item() * J
    Kk = dk[2].item() * K

    Kz = B0dir[0].item() * Ki + B0dir[1].item() * Kj + B0dir[2].item() * Kk
    K2 = Ki**2 + Kj**2 + Kk**2

    center = K2 == 0
    K2[center] = xp.inf

    D = 1/3 - Kz**2 / K2
    D[center] = DCoffset
    return fftshift(D).astype(xp.float32)


def simulate_RDF_ppm(chi_ppm, voxelSize_mm, B0dir, zeropadding=True):
    chi_ppm = move2gpu(chi_ppm)
    matrixSize = chi_ppm.shape
    if zeropadding:
        padsize = tuple(np.ceil((x+1)/2).astype(int) for x in matrixSize)
        kernelSize = tuple(a+2*b for a,b in zip(matrixSize, padsize))
    else:
        kernelSize = matrixSize

    D = get_dipoleKernel_kspace(kernelSize, voxelSize_mm, B0dir)

    if zeropadding:
        chiBig = np.pad(chi_ppm, ((padsize[0], padsize[0]), \
                                  (padsize[1], padsize[1]), \
                                  (padsize[2], padsize[2])), 'constant', \
                        constant_values = 0)

        psi = depad_array3d(ifftn(D * fftn(chiBig)).real, padsize)
    else:
        psi = ifftn(D * fftn(chi_ppm)).real
    return move2cpu(psi).astype(xp.float32)



def loadv73matlab(path2file):
    with h5.File(path2file, 'r') as f:
        attrs_dict = recursively_load_attrs(f)
        data_dict = recursively_load_data(f, attrs_dict)
    return nest_dict(data_dict)


def recursively_load_attrs(h5file, path='/'):
    """
    recursively load attributes for all groups and datasets in
    hdf5 file as python dict
    :param h5file: h5py.File(<filename>, 'r')
    :param path: "directory path" in h5 File
    :returns:
    :rtype: nested dicts
    """

    attrs_dict = {}
    for k, v in h5file[path].items():

        d = {}
        for ak, av in v.attrs.items():
            d[ak] = av

        if isinstance(v, h5._hl.dataset.Dataset):
            attrs_dict[k] = d

        elif isinstance(v, h5._hl.group.Group):
            d.update(recursively_load_attrs(
                h5file, os.path.join(path, k)))
            attrs_dict[k] = d

    return attrs_dict


def recursively_load_data(h5file, attrs_dict, path='/'):
    """
    recursively load data for all groups and datasets in
    hdf5 file as python dict corresponding to attrs_dict
    (see function recursively_load_attrs)
    :param h5file: h5py.File(<filename>, 'r')
    :param attrs_dict: output of function recursively_load_attrs
    :returns:
    :rtype: nested dicts
    """

    result = {}
    for k, v in attrs_dict.items():

        if k == '#refs#':
            continue

        if k == '#subsystem#':
            continue

        if isinstance(v, dict):

            if v.get('MATLAB_class') == b'function_handle':
                continue
            elif v.get('MATLAB_class') != b'struct':

                val = h5file[path+k+'/'][...]
                arrays3d = np.array(['signal', 'refsignal', 'fieldmap_Hz', 'R2s_Hz',
                                     'water', 'fat', 'silicone', 'fatFraction_percent'])
                if ~np.isin(k, arrays3d):
                    val = np.squeeze(val)

                if isinstance(val, np.ndarray) and \
                    val.dtype == [('real', '<f4'), ('imag', '<f4')]:
                    val = np.transpose(val.view(np.complex64)).astype(np.complex64)
                elif isinstance(val, np.ndarray) and \
                    (val.dtype == [('real', '<f8'), ('imag', '<f8')]):
                    val = np.transpose(val.view(np.complex128)).astype(np.complex64)
                elif isinstance(val, np.ndarray) and \
                    (val.dtype == 'float64' or val.dtype == 'float32'):
                    val = (np.transpose(val).astype(np.float32))
                elif isinstance(val, np.ndarray) and \
                   (val.dtype == 'uint64' or val.dtype == 'uint32'):
                    val = (np.transpose(val).astype(np.uint32))
                elif isinstance(val, np.ndarray) and \
                   (val.dtype == 'bool_' or val.dtype == 'uint8'):
                    val = (np.transpose(val).astype(np.bool_))

                if v.get('MATLAB_class') == b'char':
                    try:
                        val = ''.join([chr(c) for c in val])
                    except:
                        val = ''

                result[path+k+'/'] = val
            else:
                result.update(recursively_load_data(h5file, v, path+k+'/'))

    return result

def nest_dict(flat_dict):
    seperator = '/'
    nested_dict = {}
    for k, v in flat_dict.items():

        path_list = list(filter(None, k.split(seperator))) # removes '' elements
        split_key = path_list.pop(0)
        left_key = seperator.join(path_list)

        if left_key == '':
            nested_dict[split_key] = v
            continue

        if not nested_dict.get(split_key): # init new dict
            nested_dict[split_key] = {}

        if left_key != '':
            nested_dict[split_key].update({left_key: v})

    return nested_dict
