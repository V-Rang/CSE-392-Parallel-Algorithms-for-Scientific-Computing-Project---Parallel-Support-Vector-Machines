#include "cuda.h"
#include <iostream>
#include <vector>
#include <math.h>
using namespace std;
typedef vector<float> vf;
#define BSIZE 32
#include <limits>



void printvctr(vf &a)
{
    cout<<"printing vctr"<<endl;
    for (auto i : a)
    {
        cout<<i<<", ";
    }
}

void printmtx(vf &amat)
{
    
}

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

 __global__ void multiply(int M, int N, int K,   float* __restrict__ A, float* __restrict__ B, float *C) 
 {

  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;


  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    
    C[x * N + y] = tmp;
  }
  }


 __global__ void multiplycoal(int M, int N, int K,   float* __restrict__ A, float* __restrict__ B, float *C) 
 {

 const unsigned int  x = blockIdx.x * BSIZE + (threadIdx.x / BSIZE);//for coalescing
 const unsigned int  y = blockIdx.y * BSIZE + (threadIdx.x % BSIZE);



  if (x < M && y < N) {
    float tmp = 0.0;
    #pragma unroll
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    
    C[x * N + y] = tmp;
  }
  }


__global__ void shmultiply(float* __restrict__ A, float* __restrict__ B, float *C, int N) {

    int i,j;
    float temp = 0;
   // float temp2=0;

    __shared__ float As [BSIZE][BSIZE];
    __shared__ float Bs [BSIZE][BSIZE];

    int tidx=threadIdx.x;
    int tidy=threadIdx.y;
    // Row i of matrix left
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;


    for (int nTile = 0; nTile < gridDim.x; nTile++) {

        j = nTile * BSIZE + threadIdx.x;
        i = nTile * BSIZE + threadIdx.y;
      

        As[tidy][tidx] = A[row * N + j];

        Bs[tidy][tidx] = B[i * N + col]; // Coalesced access
        __syncthreads();
   //    #pragma unroll
        for (int k = 0; k < BSIZE; k+=2) {//unroll a bit manually ffor performance?

            temp += As[tidy][k] * Bs[k][tidx]; 
            temp += As[tidy][k+1] * Bs[k+1][tidx];
        }
        // Synchronize
        __syncthreads();
    }
    C[row * N + col] = temp;
}


int main()
{
    //A_m,k B_k,n =C_m,n
    int M=4096;
    int N=M;
    int K=N;
    vf a(M*K,0.0);
    vf b(K*N,0.0);
    vf c(M*N,0.0);
    vf chost(M*N,0.0);
    float *da,*db,*dc;
    for(int i=0;i<a.size();i++)
    {
        a[i]=i+1;
        a[i]/=M;
    }
    for(int i=0;i<b.size();i++)
    {
        b[i]=b.size()-i;
        b[i]/=M;
    }
    
    
    int kk=ceil(100/32);
    int siz=a.size()*sizeof(float);
    cudaMalloc((void**)&da, siz);
    cudaMalloc((void**)&db, siz);
    cudaMalloc((void**)&dc, siz);

    cudaMemcpy(da, a.data(), siz, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b.data(), siz, cudaMemcpyHostToDevice);
    //cudaMemcpy(dc, c.data(), siz, cudaMemcpyHostToDevice);


cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
   
   cudaEventRecord(start);
    //dim3 blockDim(bsize * bsize);
//normal mult grid dim bdim
int bsize=32;
  // dim3 blockDim(bsize , bsize);
  // dim3 gridDim(ceil(M/bsize)+1, ceil(N/bsize)+1);
//coal dims
   dim3 blockDim(bsize * bsize,1);
   dim3 gridDim(ceil(M/bsize)+1, ceil(N/bsize)+1);

//shared mult grid dim bdim
  // int bsize=BSIZE;
  // dim3 blockDim(bsize , bsize);
  // dim3 gridDim(ceil(M/bsize)+1, ceil(N/bsize)+1);
//naive mm
  // multiply<<<gridDim,blockDim>>>(M,N,K,da,db,dc);
//coalesced mm
   multiplycoal<<<gridDim,blockDim>>>(M,N,K,da,db,dc);
//sharedmm
//   shmultiply<<<gridDim,blockDim>>>(da,db,dc,M);



cudaEventRecord(stop);
cudaEventSynchronize(stop);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
cout<<milliseconds<<endl;


cudaMemcpy(c.data(), dc, siz, cudaMemcpyDeviceToHost);


  //  printvctr(c);


int hostchk=0;
if(hostchk)
{
    float tmp=0;
    for(int i =0;i<M;i++)
    {
        for(int j=0;j<N;j++)
        {
            tmp=0;
            for(int k=0;k<K;k++)
            {
                tmp+=a[K*i+k]*b[N*k+j];
               // cout<<K*i+k<<endl;
            }
            chost[M*i+j]=tmp;
        }
    }


for( int i=0;i<M;i++)
{
    for(int j=0;j<N;j++)
    {
        if (abs(chost[M*i+j]-c[M*i+j])>1)
        cout<<chost[M*i+j]<<"  "<<c[M*i+j]<<endl;
    }
}
//printvctr(chost);
}


    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);


//cuda_hello<<<1,1>>>();
cudaDeviceSynchronize();




    


return 0;

    
    
}
