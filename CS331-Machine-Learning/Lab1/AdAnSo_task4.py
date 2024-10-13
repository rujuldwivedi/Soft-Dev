'''
Team: AdAnSo
Members:
1. Adarsh Anand (2003101)
2. Aniket Chaudhri (2003104)
3. Somesh Agrawal (2003326)
'''

# python file to import matrix multiplication written in c++ boost and call from here, matrix available in txt files

import matrixboost
import numpy as np
import time

# read matrix from txt file
def read_matrix(filename):
    matrix = np.loadtxt(filename)
    return matrix

# write matrix to txt file
def write_matrix(filename, matrix):
    np.savetxt(filename, matrix)


# main function
def main():
    # read matrix from txt file
    matrix1 = read_matrix('matrix1.txt').astype(np.int32)
    matrix2 = read_matrix('matrix2.txt').astype(np.int32)

    starttime = time.time()

    # call matrix multiplication function from c++ boost
    matrixboost.matrixmul(matrix1, matrix2)

    endtime = time.time()
    print('Time taken: ', endtime - starttime, 'seconds')


if __name__ == '__main__':
    main()

