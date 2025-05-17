import numba
import numpy as np
from lib import CudaProblem, Coord
from typing import Callable
import warnings

warnings.filterwarnings(
    action="ignore", category=numba.NumbaPerformanceWarning, module="numba"
)


def dot_spec(a, b):
    return a @ b


TPB = 8


def dot_test(cuda):
    def call(out, a, b, size) -> None:
        shared = cuda.shared.array(TPB, numba.float32)

        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        local_i = cuda.threadIdx.x
        # FILL ME IN (roughly 9 lines)
        if i < size:
            shared[local_i] = a[i] * b [i]
            cuda.syncthreads()
        if i > 0:
            return 
        acc = 0
        for idx in range(size):
            acc += shared[idx]
        out[0] = acc

    return call


SIZE = TPB
out = np.zeros(1)
a = np.arange(SIZE)
b = np.arange(SIZE)
problem = CudaProblem(
    "Dot",
    dot_test,
    [a, b],
    out,
    [SIZE],
    threadsperblock=Coord(TPB, 1),
    blockspergrid=Coord(1, 1),
    spec=dot_spec,
)
problem.show()
problem.check()