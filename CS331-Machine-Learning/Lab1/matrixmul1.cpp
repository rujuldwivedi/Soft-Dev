#include <bits/stdc++.h>
#define BOOST_BIND_GLOBAL_PLACEHOLDERS
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <string>

namespace np = boost::python::numpy;

using namespace std;

vector<vector<int>> vector_convertor(np::ndarray a) {
    int n = a.shape(0);
    int m = a.shape(1);
    vector<vector<int>> B(n, vector<int>(m, 0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            B[i][j] = boost::python::extract<int>(a[i][j]);
        }
    }
    return B;
}

void matrixmul(np::ndarray A, np::ndarray B) {
    vector<vector<int>> Mtrx1 = vector_convertor(A);
    vector<vector<int>> Mtrx2 = vector_convertor(B);
    vector<vector<int>> mult;

    for (int i = 0; i < Mtrx1.size(); ++i) {
        vector<int> v;
        for (int j = 0; j < Mtrx2[0].size(); ++j) {
            int temp = 0;
            for (int k = 0; k < Mtrx1[0].size(); ++k) {
                temp += Mtrx1[i][k] * Mtrx2[k][j];
            }
            v.push_back(temp);
        }
        mult.push_back(v);
    }

    // Displaying the multiplication of two matrix.
    // cout << endl << "Output Matrix: " << endl;
    // for(int i = 0; i < r1; ++i)
    // for(int j = 0; j < c2; ++j)
    // {
    //     cout << " " << mult[i][j];
    //     if(j == c2-1)
    //         cout << endl;
    // }

    // return mult;
}

BOOST_PYTHON_MODULE(matrixmul1) {
    using namespace boost::python;
    np::initialize();
    def("matrixmul", matrixmul);
}
