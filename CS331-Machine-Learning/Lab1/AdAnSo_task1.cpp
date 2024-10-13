/*
Team: AdAnSo
Members:
1. Adarsh Anand (2003101)
2. Aniket Chaudhri (2003104)
3. Somesh Agrawal (2003326)
*/

#include<bits/stdc++.h>

using namespace std;

#define n 600

void matrixmul(vector<vector<int>> &A, vector<vector<int>> &B,
               vector<vector<int>> &C) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0;
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main()
{
    
    vector<vector<int>> A(n, vector<int>(n));
    vector<vector<int>> B(n, vector<int>(n));
    vector<vector<int>> C(n, vector<int>(n));

    // Read matrix A and B from matrix1.txt and matrix2.txt

    freopen("matrix1.txt", "r", stdin);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cin >> A[i][j];
        }
    }
    // for (auto x:A){
    //     for (auto y:x){
    //         cout << y << " ";
    //     }
    // }
    freopen("matrix2.txt","r",stdin);
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<n;j++)
        {
            cin>>B[i][j];
        }
    }
    // for (auto x:B){
    //     for (auto y:x){
    //         cout << y << " ";
    //     }
    // }
    // note time
    auto start = chrono::high_resolution_clock::now();
    matrixmul(A,B,C);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "Time taken by function: "
         << duration.count()/1000000.0 << " seconds" << endl;
    // freopen("matrix3.txt","w",stdout);
    // for(int i=0;i<n;i++)
    // {
    //     for(int j=0;j<n;j++)
    //     {
    //         cout<<C[i][j]<<" ";
    //     }
    //     cout<<endl;
    // }
    return 0;
}
