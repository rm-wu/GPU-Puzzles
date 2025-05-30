import numba
import numpy as np
from lib import CudaProblem, Coord
from typing import Callable
import warnings

warnings.filterwarnings(
    action="ignore", category=numba.NumbaPerformanceWarning, module="numba"
)


def pool_spec(a):
    out = np.zeros(*a.shape)
    for i in range(a.shape[0]):
        out[i] = a[max(i - 2, 0) : i + 1].sum()
    return out


TPB = 8


def pool_test(cuda) -> Callable:
    def call(out, a, size) -> None:
        shared = cuda.shared.array(TPB, numba.float32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        local_i = cuda.threadIdx.x
        # FILL ME IN (roughly 8 lines)
        if i < size:
            shared[local_i] = a[i]
            cuda.syncthreads()
        else:
            return
        if i == 0:
            out[i] = shared[local_i]
        elif i == 1:
            out[i] = shared[local_i] + shared[local_i - 1]
        else:
            out[i] = shared[local_i] + shared[local_i - 1] + shared[local_i - 2]

    return call


SIZE = 8
out = np.zeros(SIZE)
a = np.arange(SIZE)
problem = CudaProblem(
    "Pooling",
    pool_test,
    [a],
    out,
    [SIZE],
    threadsperblock=Coord(TPB, 1),
    blockspergrid=Coord(1, 1),
    spec=pool_spec,
)
problem.show()
problem.check()
