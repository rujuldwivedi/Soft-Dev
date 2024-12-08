#include<bits/stdc++.h>
#include<boost/python.hpp>
using namespace std;

namespace bp = boost::python;

//Function to multiply two matrices
void multiply(bp::list a,bp::list b)
{
    // int m = b.shape[1];
    int m=len(a);

    int n = len(bp::extract<bp::list>(b[0]));
    cout<<"M , N : "<<m<<"\t"<<n;
    // std::vector<std::vector<int>> c( m, std::vector<int> (n, 0));
    int c[m][n];

    for(int i=0 ; i<m ; i++)
    {
        for(int j=0 ;j<n ; j++)
            c[i][j]=0;
    }
    
    //Timer Starting Point
    time_t start, finish;
    time(&start);

    for (int i = 0; i < len(b[0]); ++i)
    {
        for (int j = 0; j < len(a[0]); ++j)
        {
            for (int k = 0; k < len(a); ++k)
            {
                c[i][j] = c[i][j] + bp::extract<int>(a[i][k]) * bp::extract<int>(b[k][j]);
            }
        }
    }

    for(int i=0 ; i<m ; i++)
    {
        for(int j=0 ;j<n ; j++)
            cout<<c[i][j]<<" ";
        cout<<"\n";
    }
    
    
    //Timer ending point
    time(&finish);
	
    cout << "Time required = " << difftime(finish, start) << " seconds\n";
}

BOOST_PYTHON_MODULE(sample2) {
    // An established convention for using boost.python.
    using namespace boost::python;

    // Expose the function hello().
    def("multiply", multiply);
}
