import numba
import numpy as np
from lib import CudaProblem, Coord
from typing import Callable
import warnings

warnings.filterwarnings(
    action="ignore", category=numba.NumbaPerformanceWarning, module="numba"
)

def matmul_spec(a, b):
    return a @ b


TPB = 3
def mm_oneblock_test(cuda): 
    def call(out, a, b, size:int) -> None:
        a_shared = cuda.shared.array((TPB, TPB), numba.float32)
        b_shared = cuda.shared.array((TPB, TPB), numba.float32)

        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

        local_i = cuda.threadIdx.x
        local_j = cuda.threadIdx.y

        
        acc = 0
        for tile_offset in range(0, size, TPB):
            # tile_offset keeps track of where we are reading in the input matrix
            # since we are performning operations in blocks of (TPB, TPB) blocks
            if i < size and local_j + tile_offset < size:
                a_shared[local_i, local_j] = a[i, local_j + tile_offset]
            if j < size and local_i + tile_offset < size:
                b_shared[local_i, local_j] = b[local_i + tile_offset, j]
            cuda.syncthreads()
            
            for c in range(min(TPB, size - tile_offset)):
                acc += (a_shared[local_i, c] * b_shared[c, local_j])
            # Write the result
        if i < size and j <size:
            out[i, j] = acc
        return
                  
    return call

# Test 1

SIZE = 2
out = np.zeros((SIZE, SIZE))
inp1 = np.arange(SIZE * SIZE).reshape((SIZE, SIZE))
inp2 = np.arange(SIZE * SIZE).reshape((SIZE, SIZE)).T

problem = CudaProblem(
    "Matmul (Simple)",
    mm_oneblock_test,
    [inp1, inp2],
    out,
    [SIZE],
    Coord(1, 1),
    Coord(TPB, TPB),
    spec=matmul_spec,
)
problem.show(sparse=True)
problem.check()

SIZE = 8
out = np.zeros((SIZE, SIZE))
inp1 = np.arange(SIZE * SIZE).reshape((SIZE, SIZE))
inp2 = np.arange(SIZE * SIZE).reshape((SIZE, SIZE)).T

problem = CudaProblem(
    "Matmul (Full)",
    mm_oneblock_test,
    [inp1, inp2],
    out,
    [SIZE],
    Coord(3, 3),
    Coord(TPB, TPB),
    spec=matmul_spec,
)
problem.show(sparse=True)