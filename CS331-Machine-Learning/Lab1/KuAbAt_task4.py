import numpy as np
import time
import matrixmul1

# getting input values for matrix 1 and matrix 2

# R1 = int(input("Enter the number of rows for matrix 1:"))
# C1 = int(input("Enter the number of columns for matrix 1:"))

# R2 = int(input("Enter the number of rows for matrix 2:"))
# C2 = int(input("Enter the number of columns for matrix 2:"))

# hardcode all above values to 5
R1 = 5
C1 = 5
R2 = 5
C2 = 5

if C1 != R2:
        print("The number of columns in Matrix-1  must be equal to the number of rows in " + "Matrix-2", end='')
        print("\n")


else:
        print("Enter the entries in a single line (separated by space): for matrix 1")
        # User input of entries in a single line separated by space
        # entries_1 = list(map(int, input().split()))

        print("Enter the entries in a single line (separated by space): for matrix 2")
        # entries_2 = list(map(int, input().split()))  

        # get entries using random numbers
        entries_1 = np.random.randint(1, 10, R1*C1)
        entries_2 = np.random.randint(1, 10, R2*C2)

    

        matrix_1 = np.array(entries_1).reshape(R1, C1).astype(np.int32)
        matrix_2 = np.array(entries_2).reshape(R2, C2).astype(np.int32)

        print("result")
        print("\n")
        
        start=time.time()
        
        matrixmul1.matrixmul(matrix_1, matrix_2)
        
        
        end = time.time()
        
        print( end-start)