#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import print_function

import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import numpy as np
import pycuda.driver as cuda
import pycuda.tools
import pycuda.autoinit
import numpy as np
from pycuda.compiler import SourceModule
import pycuda.cumath
import math


# In[ ]:


mod3 = SourceModule("""
    __global__ void multiplyyd(int M, int N, int K,   double *A,
                             double *B, double *C) {
  // compute position in C that this thread is responsible for
  int BLOCKSIZE=32;
const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  // `if` condition is necessary for when M or N aren't multiples of 32.
  if (x < M && y < N) {
    double tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    
    C[x * N + y] = tmp;
  }
  }
""")


# In[ ]:


import math
from skcuda import cublas
import skcuda.cusolver as solver
import skcuda.linalg as culinalg
import math
from skcuda import misc
from pycuda import curandom
culinalg.init()
rand = curandom.MRG32k3aRandomNumberGenerator()
func = mod3.get_function("multiplyyd")
handle = misc._global_cublas_handle

typ=np.float64
a=np.loadtxt('LSSVM/result.txt')
a=typ(a)
bvec=np.loadtxt('LSSVM/bvec.txt')
bvec=typ(bvec)
m,k=a.shape
n=math.ceil(k*0.4)
m=np.int32(m)
n=np.int32(n)
k=np.int32(k)



aa=a.reshape(a.shape[0]*a.shape[1],1).astype(typ)
a_gpu = gpuarray.to_gpu(aa)
b_gpu=gpuarray.to_gpu(bvec)

#b_gpu=b_gpu.reshape(b_gpu.shape[0],1)
start = cuda.Event() 
end = cuda.Event()
start.record()
#bb=np.ones(k*n).astype(typ)
Omega = gpuarray.ones((n,k), typ)
#rand.fill_uniform(Omega)

#cc=np.zeros(m*n).astype(typ)
Y=gpuarray.empty((m*n), typ)
bsize=1024

gsize1=math.ceil(m/32)
gsize2=math.ceil(n/32)




func(m,n,k,a_gpu,Omega.gpudata,Y, block=(bsize,1, 1), grid=(gsize1, gsize2,1))
#Y=culinalg.dot(a_gpu,Omega)



Y=Y.reshape(m,n)

YY=gpuarray.to_gpu(Y.get())
#QQq,rq=np.linalg.qr(Y.get())
Q1=culinalg.qr(YY,'economic')
#print(np.dot(QQq,rq))
#Q1,_=np.linalg.qr(Y.get())
#print(QQq)
#print('upis qq ')
#print('dw is q')
#print(Q1)


Qt=culinalg.transpose(Q1)
B = gpuarray.empty((n*k), typ)


gsize1=math.ceil(n/32)
gsize2=math.ceil(k/32)

func(n,k,m,Qt.gpudata,a_gpu,B, block=(bsize,1, 1), grid=(gsize1, gsize2,1))
#print(Qt.shape)
#print(a_gpu.shape)
#B=culinalg.dot(Qt,a_gpu.reshape(m,k))
#print(B)
B=B.reshape(n,k)


u_til, s_gpu, vh_gpu = culinalg.svd(B, 'S', 'S')
u=gpuarray.empty((m*n), typ)
gsize1=math.ceil(m/32)
gsize2=math.ceil(n/32)

func(m,n,n,Q1,u_til.gpudata,u, block=(bsize,1, 1), grid=(gsize1, gsize2,1))
u=u.reshape(m,n)

utt=culinalg.transpose(u)
v=culinalg.transpose(vh_gpu)
sol=culinalg.dot(utt,b_gpu)
#print(vh_gpu.shape)

sol=culinalg.multiply(1/s_gpu,sol)
#print(sol.shape)
sol = culinalg.dot(v,sol)
end.record() 
end.synchronize() 
#print(Y)
events_secs = start.time_till(end)*1e-3
print('time total = ',events_secs)
np.savetxt('sol.txt',sol.get())



