import numpy as np
import time

from numpy.random import Generator, PCG64

def main():
	N = 1000

	#A = np.ones(N, dtype=np.float32)
	#B = np.ones(N, dtype=np.float32)

	A = np.random.rand(N,N)
	B = np.random.rand(N,N)

	start = time.time()
	C = np.add(A,B)

	print ( "C[:5] = " + str(C[:5]) )
	print ( "C[-5:] = " + str(C[-5:]) )

	vector_add_time = time.time() - start

	print ("VectorAdd took % s seconds" % vector_add_time)

	# My program takes about 0.01 seconds with a randomly generated array of floats between 0 and 1


if __name__=='__main__':
    main()
