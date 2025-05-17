import numpy as np
from lib import CudaProblem, Coord
from typing import Callable

# Problem 1: implement Map
def map_spec(a):
    return a + 10


# puzzle_3: Guards
def map_guard_test(cuda) -> Callable:
    def call(out, a, size) -> None:
        local_i = cuda.threadIdx.x
        if local_i < size:
            out[local_i] = a[local_i] + 10
    return call


SIZE = 4
out = np.zeros((SIZE,))
a = np.arange(SIZE)
problem = CudaProblem(
    "Guard",
    map_guard_test,
    [a],
    out,
    [SIZE],
    threadsperblock=Coord(8, 1),
    spec=map_spec,
)
problem.show()
problem.check()