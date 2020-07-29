import numpy as np
import time

from numba import vectorize, cuda

@vectorize(['float32(float32, float32)'], target='cuda')
def VectorAdd(a, b):
    return a + b

def main():
    # Using square matrix of size 1,000
    N = 1000

    A = np.ones(N, dtype=np.float32)
    B = np.ones(N, dtype=np.float32)

    start = time.time()
    C = VectorAdd(A, B)
    vector_add_time = time.time() - start

    print ( "C[:5] = " + str(C[:5]) )
    print ( "C[-5:] = " + str(C[-5:]) )

    print ("VectorAdd took % s seconds" % vector_add_time)

    # Mine takes around 1.1 seconds for 1,000 x 1,000 array of ones

if __name__=='__main__':
    main()


