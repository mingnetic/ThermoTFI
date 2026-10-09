import os
from functools import wraps
import fnmatch
import hdf5storage
import numpy as np
import nibabel as nib

try:
    import cupy as xp

    if os.environ.get('BMRR_USE_GPU') is None:
        os.environ['BMRR_USE_GPU'] = '1'

    import cupy as cp
    from cupy.cuda.memory import OutOfMemoryError as CUDA_OutOfMemory

    def xp_function(f):
        """GPU/CPU function decorator for class methods
        The class needs to have the attribute `self.use_gpu`

        :param f: Function with keyword argument `xp=np`
        :returns: GPU optimized function

        """
        @wraps(f)
        def wrapper(*args, **kwargs):
            try:
                if args[0].use_gpu:
                    ret = f(*args, **kwargs, xp=cp)
                else:
                    ret = f(*args, **kwargs, xp=np)
                return ret
            except CUDA_OutOfMemory:
                print('CUDA_OutOfMemory: GPU mode has been ended.')
                print('CPU mode will be enabled.')
                ret = f(*args, **kwargs, xp=np)
                return ret
        return wrapper

    def move2cpu(x, xp=cp):
        """Returns a numpy array

        :param x: numpy/cupy array
        :param xp: used python module (numpy or cupy)
        :returns: numpy array

        """
        if xp == cp:
            if isinstance(x, np.ndarray):
                y = x
            elif isinstance(x, cp.ndarray):
                y = cp.asnumpy(x)
            return y
        elif xp == np:
            return x

    def move2gpu(x, xp=cp):
        """Returns a cupy array

        :param x: numpy/cupy array
        :param xp: used python module (numpy or cupy)
        :returns: cupy array

        """
        if xp == cp:
            if isinstance(x, np.ndarray):
                y = cp.asarray(x)
            elif isinstance(x, cp.ndarray):
                y = x
            return y
        elif xp == np:
            return x

    def free_mempool():
        """Clear the default cupy mempool"""
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

except ModuleNotFoundError:
    import numpy as xp

    class CUDA_OutOfMemory(Exception):
        pass

    def xp_function(f):
        """GPU/CPU function decorator

        :param f: Function with keyword argument `xp=np`
        :returns: GPU optimized function

        """
        @wraps(f)
        def wrapper(*args, **kwargs):
            ret = f(*args, **kwargs, xp=np)
            return ret
        return wrapper

    def move2cpu(x, xp=np):
        """Returns a numpy array

        :param x: numpy/cupy array
        :param xp: used python module (numpy or cupy)
        :returns: numpy array

        """
        return x

    def move2gpu(x, xp=np):
        """Returns a cupy array

        :param x: numpy/cupy array
        :param xp: used python module (numpy or cupy)
        :returns: cupy array

        """
        return x

    def free_mempool():
        """Clear the default cupy mempool"""
        pass


def files_in_path(path, string='*', recursive=False):
    files = []
    if recursive:
        for root,dirs,datein in os.walk(path):
            for file in datein:
                if fnmatch.fnmatch(file, string):
                    files.append(root + '/' + file)
    else:
        for file in sorted(os.listdir(path)):
            if fnmatch.fnmatch(file, string):
                files.append(os.path.join(path, file))
    return sorted(files)


def folders_in_path(path, string='*'):
    files = []
    for folder in next(os.walk(path))[1]:
        if fnmatch.fnmatch(folder, string):
            files.append(os.path.join(path, folder))
    return files


def get_parent_dir(path, iter=1):
    for i in range(iter):
        path = os.path.dirname(path)
    return path


def scale_array2interval_along1D(array, intervalVector, xp=xp):
    array = move2gpu(array, xp=xp)
    intervalVector = xp.array(intervalVector).astype(np.float32)

    newValueRange = (xp.max(intervalVector) - xp.min(intervalVector))*xp.ones_like(array)

    valueRange = xp.repeat(xp.array(xp.nanmax(array, axis=-1) - xp.nanmin(array, axis=-1))[:, xp.newaxis],
                           array.shape[-1], axis=-1)

    rescaleSlope = xp.ones_like(array)
    rescaleSlope[valueRange != 0] = xp.divide(newValueRange[valueRange != 0],
                                               valueRange[valueRange != 0])

    arrayScaled = xp.multiply(rescaleSlope, array)

    # shift by intercept
    rescaleIntercept = xp.repeat(xp.array(xp.min(intervalVector) - xp.nanmin(arrayScaled, axis=-1))[:, xp.newaxis],
                                 array.shape[-1], axis=-1)
    arrayScaled += rescaleIntercept

    return move2cpu(arrayScaled, xp)


def scale_array2interval(array, intervalVector, xp=xp):
    array = move2gpu(array, xp)
    intervalVector = xp.array(intervalVector).astype(np.float32)
    newValueRange = xp.max(intervalVector) - xp.min(intervalVector)
    valueRange = xp.max(array) - xp.min(array)

    if valueRange == 0:
        rescaleSlope = 1
    else:
        rescaleSlope = newValueRange / valueRange

    arrayScaled = rescaleSlope * array

    # shift by intercept
    rescaleIntercept = xp.min(intervalVector) - xp.min(arrayScaled)
    arrayScaled += rescaleIntercept

    return move2cpu(arrayScaled, xp)


def load_nii_array(path2file):
    img = nib.load(path2file)
    img = img.get_fdata()
    return np.flip(np.transpose(img), 0)
