// matrix multiplication using C++ boost

#include <iostream>
#include <vector>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include<bits/stdc++.h>

namespace np = boost::python::numpy;

using namespace std;

// // ndarray to vector
std::vector<std::vector<int>> ndarray_to_vector(np::ndarray A) {
    int n = A.shape(0);
    int m = A.shape(1);
    std::vector<std::vector<int>> B(n, std::vector<int>(m, 0));
    for (int i = 0; i < n; i++){
        for (int j = 0; j < m; j++){
            B[i][j] = boost::python::extract<int>(A[i][j]);
        }
    }
    return B;
}

// matrix multiplication function
void matrixmul(np::ndarray A, np::ndarray B){
    std::vector<std::vector<int>> nA = ndarray_to_vector(A);
    std::vector<std::vector<int>> nB = ndarray_to_vector(B);
    int n = nA.size();
    int m = nB[0].size();
    int p = nB.size();
    std::vector<std::vector<int>> C(n, std::vector<int>(m, 0));
    for (int i = 0; i < n; i++){
        for (int j = 0; j < m; j++){
            for (int k = 0; k < p; k++){
                C[i][j] += nA[i][k] * nB[k][j];
            }
        }
    }
    // std::cout << "The product of the two matrices is: \n";
    // for (int i = 0; i < n; i++){
    //     for (int j = 0; j < m; j++){
    //         std::cout << C[i][j] << " ";
    //     }
    //     std::cout << '\n';
    // }
    // return C;

}

BOOST_PYTHON_MODULE(matrixboost)
{
    using namespace boost::python;
    np::initialize();
    def("matrixmul", matrixmul);
    
}


