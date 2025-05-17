import numpy as np
from lib import CudaProblem, Coord
from typing import Callable


# Problem 1: implement Map
def map_spec(a):
    return a + 10


def map_test(cuda) -> Callable:
    def call(out, a) -> None:
        local_i = cuda.threadIdx.x
        out[local_i] = a[local_i] + 10
        return 
 
    return call


SIZE = 4
out = np.zeros((SIZE,))
a = np.arange(SIZE)


problem = CudaProblem("Map", map_test, [a], out, threadsperblock=Coord(SIZE, 1), spec=map_spec)
problem.show()
problem.check()

