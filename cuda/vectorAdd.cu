// This program computes the sum of two vectors on the GPU using CUDA
// By: Roland Green

#include <cassert>
#include <cstdlib>
#include <iostream>

using std::cout;
using std::endl;

// Vector Addition kernel
__global__ void vectorAdd(int *a, int *b, int *c, int N) {
  // Global threadID calculation
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Boundary check
  if (tid < N) c[tid] = a[tid] + b[tid];
}

int main() {
  // Size of our arrays
  int N = 1 << 10;
  size_t bytes = N * sizeof(int);

  // Host pointers
  int *h_a, *h_b, *h_c;
  h_a = new int[N];
  h_b = new int[N];
  h_c = new int[N];

  // Initialize data
  for (int i = 0; i < N; i++) {
    h_a[i] = rand() % 100;
    h_b[i] = rand() % 100;
  }

  // Device pointers
  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  // Copy data to the device
  cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

  // Set CTA and Grid dimensions
  int THREADS = 1024;
  int BLOCKS = N / THREADS;

  // Launch the kernel
  vectorAdd<<<BLOCKS, THREADS>>>(d_a, d_b, d_c, N);

  // Copy the data back
  cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

  // Functional test
  for (int i = 0; i < N; i++) {
    assert(h_c[i] == h_a[i] + h_b[i]);
  }

  cout << "COMPLETED SUCCESSFULLY!" << endl;

  return 0;
}
