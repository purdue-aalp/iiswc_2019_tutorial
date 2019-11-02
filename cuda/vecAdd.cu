__global__ void vectorAdd(int *a, int *b, int *c){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	c[tid] = a[tid] + b[tid];
}

int main() {
	int N = 1 << 10;
	size_t bytes = N * sizeof(int);

	int *h_a, *h_b, *h_c;
	h_a = new int[N];
	h_b = new int[N];
	h_c = new int[N];
	for( int i = 0; i < N; i++ ){
		h_a[i] = 1;
		h_b[i] = 2;
	}

	int *d_a, *d_b, *d_c;
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);

	cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

	vectorAdd<<<N / 128, 128>>>(d_a, d_b, d_c);

	cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

	return 0;
}
