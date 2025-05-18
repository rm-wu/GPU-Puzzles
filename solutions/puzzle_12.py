import numba
import numpy as np
from lib import CudaProblem, Coord
from typing import Callable
import warnings

warnings.filterwarnings(
    action="ignore", category=numba.NumbaPerformanceWarning, module="numba"
)

# Constants
TPB = 8  # Threads per block (must be power of 2 for efficient tree reduction)

def sum_spec(a):
    """
    Reference implementation for block-wise sum reduction.
    Computes sum for each block of TPB elements.
    """
    out = np.zeros((a.shape[0] + TPB - 1) // TPB)
    for j, i in enumerate(range(0, a.shape[-1], TPB)):
        out[j] = a[i : i + TPB].sum()
    return out

def sum_test(cuda) -> Callable:
    def call(out, a, size: int) -> None:
        # Allocate shared memory for block-level reduction
        # Each thread will load one element into shared memory
        cache = cuda.shared.array(TPB, numba.float32)
        
        # Calculate global and local thread indices
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x  # Global index
        local_i = cuda.threadIdx.x  # Local index within block

        # Phase 1: Load data into shared memory
        # Each thread loads one element if within bounds
        if i < size:
            cache[local_i] = a[i]
        else:
            return  # Threads beyond array bounds exit early
        cuda.syncthreads()  # Ensure all threads have loaded their data

        # Phase 2: Tree-based reduction
        # This implements a parallel reduction pattern where:
        # - Each iteration combines pairs of elements
        # - The step size doubles each iteration
        # - Only threads at even multiples of step participate
        # Example for TPB=8:
        # Iteration 1 (step=1): [0,1,2,3,4,5,6,7] -> [0+1,2+3,4+5,6+7,_,_,_,_]
        # Iteration 2 (step=2): [0+1,2+3,4+5,6+7,_,_,_,_] -> [(0+1)+(2+3),(4+5)+(6+7),_,_,_,_,_,_]
        # Iteration 3 (step=4): [(0+1)+(2+3),(4+5)+(6+7),_,_,_,_,_,_] -> [sum of all elements,_,_,_,_,_,_,_]
        step = 1
        while step < TPB:
            if local_i % (2 * step) == 0:  # Only threads at even multiples of step participate
                if local_i + step < TPB:
                    cache[local_i] = cache[local_i] + cache[local_i + step]
                step *= 2  # Double the step size for next iteration
            else:
                return  # Other threads exit as they're not needed
            cuda.syncthreads()  # Ensure all additions are complete before next iteration

        # Phase 3: Write result
        # Only thread 0 writes the final sum for this block
        if local_i == 0:
            out[cuda.blockIdx.x] = cache[0]

    return call

def run_tests():
    # Test 1: Simple case - single block, power of 2 size
    # Tests basic functionality with minimal edge cases
    SIZE = 8
    out = np.zeros(1)
    inp = np.arange(SIZE)  # Simple sequence: [0,1,2,3,4,5,6,7]
    problem = CudaProblem(
        "Sum (Simple)",
        sum_test,
        [inp],
        out,
        [SIZE],
        Coord(1, 1),  # Single block
        Coord(TPB, 1),
        spec=sum_spec,
    )
    problem.show()
    problem.check()

    # Test 2: Multiple blocks case
    # Tests handling of multiple blocks and non-power-of-2 size
    SIZE = 15
    out = np.zeros(2)  # Two blocks needed
    inp = np.arange(SIZE)  # Sequence: [0,1,2,...,14]
    problem = CudaProblem(
        "Sum (Full)",
        sum_test,
        [inp],
        out,
        [SIZE],
        Coord(2, 1),  # Two blocks
        Coord(TPB, 1),
        spec=sum_spec,
    )
    problem.show()
    problem.check()

    # Test 3: Complex case - large input with pattern
    # Tests multiple blocks, complex patterns, and edge cases
    SIZE = 32  # Large size to test multiple blocks
    out = np.zeros(4)  # Four blocks needed
    # Create a complex input pattern: i * (i % 3 + 1)
    # This creates a sequence with varying differences between elements
    # Pattern: 0,1,4,3,8,5,12,7,16,9,20,11,24,13,28,15,...
    inp = np.array([i * (i % 3 + 1) for i in range(SIZE)])
    problem = CudaProblem(
        "Sum (Complex)",
        sum_test,
        [inp],
        out,
        [SIZE],
        Coord(4, 1),  # Four blocks needed for 32 elements with TPB=8
        Coord(TPB, 1),
        spec=sum_spec,
    )
    problem.show()
    problem.check()
    
    # My comment: additional test to check that the code generalizes to
    # 3 blocks or more
    SIZE = 23
    out = np.zeros((SIZE - 1) // TPB + 1)
    inp = np.arange(SIZE)
    problem = CudaProblem(
        "Sum (Full)",
        sum_test,
        [inp],
        out,
        [SIZE],
        Coord((SIZE - 1) // TPB + 1, 1),
        Coord(TPB, 1),
        spec=sum_spec,
    )
    problem.check()
    problem.show()
    

if __name__ == "__main__":
    run_tests()
