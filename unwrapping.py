from skimage.restoration import unwrap_phase
#from bmrr_wrapper.Backend.qsm.helper import laplacian_unwrap
import os
import matlab

import numpy as np
try:
    import cupy as xp
    import cupy as cp
except ModuleNotFoundError:
    import numpy as xp


def UnwrappingWrapper(inPhase, mask, method, voxelSize_mm):
    assert np.min(inPhase) >= -2 * np.pi and np.max(inPhase) <= 2 * np.pi
    if mask is None:
        mask = np.ones(inPhase.shape, dtype=np.bool)

    if method == 'skimage':
        outPhase = unwrap_phase(np.ma.array(inPhase, mask=~mask))
        outPhase = np.asarray(outPhase)
    elif method == 'qgu':
        outPhase = QualityGuidedUnwrapping(inPhase, mask)

    return (outPhase * mask).astype(np.float32)


def QualityGuidedUnwrapping(inPhase, mask):
    import matlab.engine
    eng = matlab.engine.start_matlab()
    path = os.path.dirname(os.path.abspath(__file__))
    path += '/Quality_guided_unwrapping'
    eng.addpath(path)

    unwrapped = eng.qualityGuidedUnwrapping(
        matlab.double(inPhase.tolist()),
        matlab.logical(mask.tolist()), 3.5, nargout=1
    )

    return MatlabEngineArray2numpyArray(unwrapped)

#def MatlabEngineArray2numpyArray(inArr):
#    if 'mlarray' in str(type(inArr)):
#        retArr = np.array(inArr._data).reshape(inArr.size, order='F')
#    elif isinstance(inArr, np.ndarray):
#        retArr = inArr
#    return retArr.astype(np.float32)
#import numpy as np

def MatlabEngineArray2numpyArray(inArr):
    if isinstance(inArr, np.ndarray):
        retArr = inArr
    elif isinstance(inArr, matlab.double):
        retArr = np.array(inArr._data).reshape(inArr.size, order='F')
    elif 'mlarray' in str(type(inArr)):
        retArr = np.array(inArr._data).reshape(inArr.size, order='F')
    else:
        raise TypeError(f"Unsupported input type: {type(inArr)}")

    return retArr.astype(np.float32)