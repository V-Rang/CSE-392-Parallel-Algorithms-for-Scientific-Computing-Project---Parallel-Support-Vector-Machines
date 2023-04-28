//C++ file to read a matrix A from a text file and perform the operations as given in python as:

//rank = math.ceil(0.25*Aa.shape[1])
//Omega = np.random.randn(Aa.shape[1], rank)
//Y = Aa @ Omega
//Q, _ = np.linalg.qr(Y)
//Bb = Q.T @ Aa
//u_tilde, s, v = np.linalg.svd(Bb, full_matrices = 0)
//u = Q @ u_tilde 
//solution = np.dot(u.T, B)
//solution=np.divide(solution,s)
//solution=np.dot(v.T,solution)

//The goal is to achieve faster performance with C++ code using CBLAS and LAPACKE as compared to the 
//above python implementation.

//called using icc -o executable my_mkl.cpp -mkl and timed using time ./executable

#include<iostream>
#include<mkl.h>
#include<mkl_lapacke.h> 
#include<random>
#include <fstream>

// #include"lapack.h"
// #include<cblas.h>
using namespace std;

int main()
{
    std::mt19937_64 rnd;
    std::uniform_real_distribution<double>doubleDist(0,1);

    std::ifstream in;
    in.open("A_array.txt");
    // double m;
    double element;

    // in >> element;
    // md = element;
    double* A;
    // int m = int(md);
    int m = 10001;
    A = (double *)mkl_malloc(m*m*sizeof(double),64);

    if(in.is_open())
    {
        int i=0;
        while(in >> element)
        {
            A[i++] = element;
        }
    }
    in.close();

    double *B;
    B = (double *)mkl_malloc(m*sizeof(double),64);
    B[0] = 0;
    for(int i=1;i<m;i++)
    {
        B[i] = 1;
    }


    double *Omega,*C,*tau,*work,*D;
    int k, i, j;
    k = 0.75*m;
    int info;
    int lwork;
    int n = m;
   

    // A = (double *)mkl_malloc(m*k*sizeof(double),64);
    // B = (double*)mkl_malloc(m*sizeof(double),64);

    Omega = (double *)mkl_malloc(k*n*sizeof(double),64);
    C = (double *)mkl_malloc(m*n*sizeof(double),64);
    D = (double *)mkl_malloc(m*n*sizeof(double),64);
    tau = (double *)mkl_malloc(n*sizeof(double),64);
    work = (double *)mkl_malloc(1*sizeof(double),64);


    // for(i = 0; i<m;i++)
    // {
    //     for(j = 0;j<k;j++)
    //     {
    //         cout << A[j+i*k] << " ";
    //     }
    //     cout << endl;
    // }
    

    // A[0] = 4,A[1] = 5, A[2] = 8,A[3] = 1, A[4] = 34,A[5] = 4, A[6] = 15,A[7] = 16, A[8] = 11, A[9] = 14,A[10] = 35, A[11] = 23;
    // B[0] = 3, B[1] = 4,B[2] = 5, B[3] = 6;


    // cout << "Matrix A" << endl;
    // for(i= 0;i<m;i++)
    // {
    //     for(j=0;j<m;j++)
    //     {
    //         cout << A[j+i*m] << " ";
    //     }
    //     cout << endl;
    // }
    // cout << endl;

    // cout << "Vector B" << endl;
    // for(i= 0;i<m;i++)
    // {
    //     cout << B[i] << endl;
    // }
    // cout << endl;


    for(j=0;j<(k *n);j++)
    {
        Omega[j] = static_cast<float>(rand())/static_cast<float>(RAND_MAX);
    }

    // cout<<"Matrix Omega"<<endl;
    // for(i= 0;i<n;i++)
    // {
    //     for(j=0;j<k;j++)
    //     {
    //         cout << Omega[j+i*k] << " ";
    //     }
    //     cout << endl;
    // }
    // cout<<endl;

    for(i=0;i<(m *n);i++)
    {
        C[i] = 0;
    }
    // cout <<endl;

    // for(i=0;i<(m*k);i++)
    // {
    //     cout << A[i]<<" ";
    // }
    // cout << endl;
    // for(i=0;i<(k*n);i++)
    // {
    //     cout << B[i]<<" ";
    // }
    // cout << endl;

    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,n,k,1,A,k,Omega,n,0,C,n);
    // cout << "original array = AXOmega"<<endl;
    // for(i= 0;i<m;i++)
    // {
        // for(j=0;j<n;j++)
        // {
    // cblas_dgemv(CblasRowMajor,CblasTrans,m,
            // cout << C[j+i*n] << " ";
        // }
        // cout << endl;
    // }
    // cout << endl;

    // DGEQRF(&m,&n,C,&m,tau,work,&lwork,&info);
    info =  LAPACKE_dgeqrf(LAPACK_ROW_MAJOR,m,n,C,n,tau);
    // cout << "info val is post dgeqrf" << info << endl;
    double *Q,*R;
    Q = (double *)mkl_malloc(m*n*sizeof(double),64);
    R = (double *)mkl_malloc(n*n*sizeof(double),64);

    // cout << "post DGEQRF"<<endl;
    // for(i= 0;i<m;i++)
    //     {
    //         for(j=0;j<n;j++)
    //         {
    //             cout << C[j+i*n] << " ";
    //         }
    //         cout << endl;
    //     }
    // cout << endl;

    for(int i=0;i<n;i++)
    {    
        for(j=0;j<n;j++)
        {
            if(j>=i)
            {
                R[j+i*n] = C[j+i*n];
            }
            else
            {
                R[j+i*n] = 0;
            }
        }
    }

    // cout << "R matrix"<<endl;
    // for(int i=0;i<n;i++)
    // {    
    //     for(j=0;j<n;j++)
    //     {
    //         cout << R[j+i*n] << " ";
    //     }
    //     cout << endl;
    // }
    // cout << endl;


    // dorgqr(&m,&n,&n,C,&m,tau,work,&lwork,&info); 
    info = LAPACKE_dorgqr(LAPACK_ROW_MAJOR,m,n,n,C,n,tau); 
    // cout << "info val is post dorgqr" << info << endl;

    // cout << "Q matrix"<<endl;
    // for(i= 0;i<m;i++)
    // {
    //     for(j=0;j<n;j++)
    //     {
    //         cout << C[j+i*n] << " ";
    //     }    

    //     cout << endl;
    // }
    // cout << endl;

    

    //QXR = A? Yes

    for(i=0;i<(m *n);i++)
    {
        D[i] = 0;
    }

    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,n,n,1,C,n,R,n,0,D,n);   
    // cout << "recomputed array"<<endl;
    // for(i= 0;i<m;i++)
    // {
    //     for(j=0;j<n;j++)
    //     {
    //         cout << D[j+i*n] << " ";
    //     }
    //     cout << endl;
    // }
    // cout << endl;

    //Q.T@A
    
    double *QTA;
    QTA = (double *)mkl_malloc(n*k*sizeof(double),64);
    for(i=0;i<(n *k);i++)
    {
        QTA[i] = 0;
    }

    cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,n,k,m,1,C,n,A,k,0,QTA,k);   
    // cout << "QTA"<<endl;
    // for(i= 0;i<n;i++)
    // {
    //     for(j=0;j<k;j++)
    //     {
    //         cout << QTA[j+i*k] << " ";
    //     }
    //     cout << endl;
    // }
    // cout << endl;


    //SVD OF QTA
    double *U,*S,*VT;
    int trunc = min(n,k);
    U = (double *)mkl_malloc(n*trunc*sizeof(double),64);
    S = (double *)mkl_malloc(trunc*sizeof(double),64);
    VT = (double *)mkl_malloc(trunc*k*sizeof(double),64);
    

    for(i=0;i<(n *trunc);i++)
    {
        U[i] = 0;
    }

    for(i=0;i<(trunc *k);i++)
    {
        VT[i] = 0;
    }

    for(i=0;i<trunc;i++)
    {
        S[i] = 0;
    }



    // info = LAPACKE_dorgqr(LAPACK_ROW_MAJOR,m,n,n,C,n,tau); 
    lwork = max(max(1, 3*trunc + max(n,k)),5*trunc);
    info = LAPACKE_dgesvd_work(LAPACK_ROW_MAJOR,'S','S',n,k,QTA,k,S,U,trunc,VT,k,work,lwork);

    // info = LAPACKE_dgesvd_work(LAPACK_ROW_MAJOR,'O','S',n,k,QTA,k,S,,trunc,VT,k,work,lwork);

    // cout << "U"<<endl;
    // for(i= 0;i<n;i++)
    // {
    //     for(j=0;j<trunc;j++)
    //     {
    //         cout << U[j+i*trunc] << " ";
    //     }
    //     cout << endl;
    // }
    // cout << endl;


    // cout << "S"<<endl;
    // for(i= 0;i<trunc;i++)
    // {
    //     cout<<S[i]<<" ";
    // }
    // cout << endl;

    // cout << "VT"<<endl;
    // for(i= 0;i<trunc;i++)
    // {
    //     for(j=0;j<k;j++)
    //     {
    //         cout << VT[j+i*k] << " ";
    //     }
    //     cout << endl;
    // }
    // cout << endl;

    // cout << "Q matrix"<<endl;
    // for(i= 0;i<m;i++)
    // {
    //     for(j=0;j<n;j++)
    //     {
    //         cout << C[j+i*n] << " ";
    //     }
    //     cout << endl;
    // }
    // cout << endl;

    double* ucalc;
    // int trunc = min(n,k);
    ucalc = (double *)mkl_malloc(m*trunc*sizeof(double),64);    

    for(i=0;i<(m *trunc);i++)
    {
        ucalc[i] = 0;
    }

    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,trunc,n,1,C,n,U,trunc,0,ucalc,trunc);

    // cout << "ucalc matrix"<<endl;
    // for(i= 0;i<m;i++)
    // {
    //     for(j=0;j<trunc;j++)
    //     {
    //         cout << ucalc[j+i*trunc] << " ";
    //     }
    //     cout << endl;
    // }
    // cout << endl;


    double* solu;
    solu = (double *)mkl_malloc(trunc*sizeof(double),64);   

    for(i=0;i<trunc;i++)
    {
        solu[i] = 0;
    }

    cblas_dgemv(CblasRowMajor,CblasTrans,m,trunc,1,ucalc,trunc,B,1,0,solu,1);
    // cout << "solution vector"<<endl;
    // for(i=0;i<trunc;i++)
    // {
    //     cout << solu[i] << endl;
    // }
    // cout <<endl;

    for(i = 0;i<trunc;i++)
    {
        solu[i] = solu[i]/S[i];
    }

    // cout << "solution post division by S"<<endl;
    // for(i=0;i<trunc;i++)
    // {
    //     cout << solu[i] << endl;
    // }
    // cout<<endl;



    double* solu2;
    solu2 = (double *)mkl_malloc(k*sizeof(double),64);
    for(i=0;i<k;i++)
    {
        solu2[i] = 0;
    }    

    // cblas_dgemv(CblasRowMajor,CblasTrans,m,trunc,1,ucalc,trunc,B,1,0,solu,1);
    cblas_dgemv(CblasRowMajor,CblasTrans,trunc,k,1,VT,k,solu,1,0,solu2,1);

    // cout<<"Solu 2 vector"<<endl;
    // for(i=0;i<k;i++)
    // {
    //     cout<<solu2[i]<<endl;
    // } 
    // cout<<endl;





    // cout<<""<<endl;
    // for(i= 0;i<k;i++)
    // {
    //     for(j=0;j<n;j++)
    //     {
    //         cout << Omega[j+i*n] << " ";
    //     }
    //     cout << endl;
    // }
    // cout<<endl;



    // for(i= 0;i<n;i++)
    // {
    //     for(j=0;j<n;j++)
    //     {
    //         cout << R[j+i*n] << " ";
    //     }
    //     cout << endl;
    // }
    // cout << endl;



    // DGEQRF(&m,&n,C,&m,tau,work,&lwork,&info);

    // lwork = work[0];
    // delete[] work;
    // work = new double[lwork];

    // DGEQRF(&m,&n,C,&m,tau,work,&lwork,&info);

    // if(info != 0)
    // {
    //     std::cerr << "its over" << std::endl;
    //     return 1;
    // }

    // double* work_gqr = new double[1];
    // lwork = -1;

    // for(i= 0;i<m;i++)
    // {
    //     for(j=0;j<n;j++)
    //     {
    //         if(j>=i)
    //         {
    //             C[j+i*n] = 0;       
    //         }
    //     }
    //     // cout << endl;
    // }

    // dorgqr(&m,&n,&n,C,&m,tau,work_gqr,&lwork,&info);

    // lwork = work_gqr[0];
    // delete[] work_gqr;
    // work_gqr = new double[lwork];

    // dorgqr(&m,&n,&n,C,&m,tau,work_gqr,&lwork,&info);


    // cout << info << endl;
    // for(i= 0;i<m;i++)
    // {
    //     for(j=0;j<n;j++)
    //     {
    //         cout << C[j+i*n] << " ";
    //     }
    //     cout << endl;
    // }

    // cout << info << endl;
    // for(i= 0;i<n;i++)
    // {
    //     cout << tau[i] << " ";
    // }
    // cout<<endl;



    // int rows = sizeof(arr)/sizeof(arr[0]);
    // int cols = sizeof(arr[0])/sizeof(double);

    // int rank = 0.75*cols;

    // double Omega[cols][rank];

    // for(int i =0 ; i<cols;i++)
    // {
    //     for(int j = 0; j<rank;j++)
    //     {
    //         Omega[i][j] = static_cast<float>(rand())/static_cast<float>(RAND_MAX);
    //     }
    // }




    // double* Omega  = new double(cols*rank);

    // for(uint i = 0; i<cols;i++)
    // {
    //     for(uint j = 0;j<rank;j++)
    //     {
    //         Omega[i*cols+j] = doubleDist(rnd);
    //     }
    // }

    // for(int i =0 ; i<cols;i++)
    // {
    //     for(int j = 0; j<rank;j++)
    //     {
    //         cout << Omega[i][j]<< " ";
    //     }
    //     cout << endl;
    // }

    // cblas_dgemm(CBlasRowMajor,CBlasNoTrans,CBlasNoTrans,rows,rank,cols,1,A,cols,B,rank,0,C,rank);


   
    

}