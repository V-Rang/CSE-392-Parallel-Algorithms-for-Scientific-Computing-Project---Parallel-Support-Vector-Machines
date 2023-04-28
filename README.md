# CSE-392-Parallel-Algorithms-for-Scientific-Computing-Project---Parallel-Support-Vector-Machines
Investigating runtimes for parallel SVMs by implementing randomized SVD and parallel mat-mat multiplication on Python, Torch, PyCUDA and C++ - CBLAS and LAPACKE.

The work comprises of using algorithms for training and predicting classification datasets. These algorithms are - LSSVM, PLSSVM and LIBSVM. These are given at the
following links:
1. LSSVM - https://github.com/RomuloDrumond/LSSVM
2. PLSSVM - https://github.com/SC-SGS/PLSSVM
3. LIBSVM - https://github.com/cjlin1/libsvm

The goal of this project is to try to achieve speedup by replacing the regular matrix inverse in the fit functions of the above 3 libraries with -:
1. Randomized SVD using CPU
2. rSVD using GPU
3. rSVD using C++ code using CBLAS and LAPACKE
4. rSVD using Torch
5. rSVD using PyCUDA

The algorithm for rSVD is taken from the paper: https://epubs.siam.org/doi/10.1137/090771806 "Finding Structure with Randomness: Probabilistic Algorithms for
Constructing Approximate Matrix Decompositions" - Halko, Martinsson, Tropp (2011).
