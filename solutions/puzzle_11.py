import numba
import numpy as np
from lib import CudaProblem, Coord
from typing import Callable
import warnings
import math

warnings.filterwarnings(
    action="ignore", category=numba.NumbaPerformanceWarning, module="numba"
)


# Reference implementation for testing
# Computes 1D convolution by sliding filter b over input array a
# For each position i, computes sum of a[i+j] * b[j] for all valid j
def conv_spec(a, b):
    out = np.zeros(*a.shape)
    len = b.shape[0]
    for i in range(a.shape[0]):
        out[i] = sum([a[i + j] * b[j] for j in range(len) if i + j < a.shape[0]])
    return out


# Constants for shared memory sizing
MAX_CONV = 4  # Maximum filter size supported
TPB = 8  # Threads per block (must be >= MAX_CONV for efficient loading)
TPB_MAX_CONV = (
    TPB + MAX_CONV - 1
)  # Total shared memory size needed for input array + extra elements


def conv_test(cuda):
    def call(out, a, b, a_size, b_size) -> None:
        # Get thread and block indices
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x  # Global thread index
        local_i = cuda.threadIdx.x  # Local thread index within block

        # Allocate shared memory arrays
        # a_shared: stores TPB elements of input array + MAX_CONV extra elements for convolution
        # b_shared: stores the filter (kernel) of size MAX_CONV
        a_shared = cuda.shared.array(TPB_MAX_CONV, numba.float32)
        b_shared = cuda.shared.array(MAX_CONV, numba.float32)

        # Phase 1: Load input array into shared memory
        # Each thread loads its corresponding element from global memory
        if i < a_size:
            a_shared[local_i] = a[i]

        # Phase 2: Load filter into shared memory
        # Only the first b_size threads load the filter elements
        # This ensures the filter is available to all threads in the block
        if local_i < b_size:
            b_shared[local_i] = b[local_i]

        # Phase 3: Load extra elements needed for convolution
        # We need to load elements that will be needed by threads at the end of the block
        # For a filter of size b_size, we need b_size-1 extra elements
        # We use threads with local_i in [b_size : 2*b_size-1] to load these elements
        # This ensures we don't waste threads and maintain good memory access patterns
        elif local_i < 2 * b_size - 1 and TPB + (i - b_size) < a_size:
            # Map local_i to shared memory index: TPB + (local_i - b_size)
            # Map i to global memory index: TPB + (i - b_size)
            a_shared[TPB + (local_i - b_size)] = a[TPB + (i - b_size)]

        # Synchronize to ensure all threads have completed loading data
        # This is crucial because threads need to access each other's loaded data
        cuda.syncthreads()

        # Phase 4: Compute convolution
        # Each thread computes one output element
        acc = 0.0
        if i < a_size:
            # For each position in the filter
            for j in range(b_size):
                # Check if we're within bounds of input array
                if i + j < a_size:
                    # Multiply and accumulate: input element * filter element
                    # Use local_i + j to access the correct element in shared memory
                    acc += a_shared[local_i + j] * b_shared[j]
            # Store result in output array
            out[i] = acc

    return call


# Test cases to verify correctness and performance
def run_tests():
    # Test 1: Simple case - small input and filter
    # Tests basic functionality with minimal edge cases
    SIZE = 6
    CONV = 3
    out = np.zeros(SIZE)
    a = np.arange(SIZE)  # Simple sequence: [0,1,2,3,4,5]
    b = np.arange(CONV)  # Simple filter: [0,1,2]
    problem = CudaProblem(
        "1D Conv (Simple)",
        conv_test,
        [a, b],
        out,
        [SIZE, CONV],
        Coord(1, 1),  # Single block
        Coord(TPB, 1),
        spec=conv_spec,
    )
    problem.show()
    problem.check()

    # Test 2: Full case - medium input and maximum filter size
    # Tests boundary conditions and maximum filter size
    out = np.zeros(15)
    a = np.arange(15)  # Sequence: [0,1,2,...,14]
    b = np.arange(4)  # Maximum filter size: [0,1,2,3]
    problem = CudaProblem(
        "1D Conv (Full)",
        conv_test,
        [a, b],
        out,
        [15, 4],
        Coord(2, 1),  # Two blocks
        Coord(TPB, 1),
        spec=conv_spec,
    )
    problem.show()
    problem.check()

    # Test 3: Complex case - large input with pattern and alternating filter
    # Tests multiple blocks, complex patterns, and edge cases
    SIZE = int(2**8)  # Large size to test multiple blocks
    CONV = 4  # Maximum filter size
    out = np.zeros(SIZE)
    # Create a complex input pattern: i * (i % 3 + 1)
    # This creates a sequence with varying differences between elements
    a = np.array([i * (i % 3 + 1) for i in range(SIZE)])
    # Alternating filter to test edge cases and numerical stability
    b = np.array([1, -1, 1, -1])
    problem = CudaProblem(
        "1D Conv (Complex)",
        conv_test,
        [a, b],
        out,
        [SIZE, CONV],
        Coord(math.ceil(SIZE // TPB), 1),  
        Coord(TPB, 1),
        spec=conv_spec,
    )
    problem.show()
    problem.check()


if __name__ == "__main__":
    run_tests()
