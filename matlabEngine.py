import numpy as np
import matlab
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
