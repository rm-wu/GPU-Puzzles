import numpy as np
from lib import CudaProblem, Coord
from typing import Callable


def zip_spec(a, b):
    return a + b

def zip_test(cuda) -> Callable:
    def call(out, a, b) -> None:
        local_i = cuda.threadIdx.x
        out[local_i] = a[local_i] + b[local_i]
        return
    
    return call

SIZE = 4
out = np.zeros((SIZE,))
a = np.arange(SIZE)
b = np.arange(SIZE)


problem = CudaProblem(
    "Zip", zip_test,  [a, b], out, threadsperblock=Coord(SIZE, 1), spec=zip_spec
)
problem.show()
problem.check()