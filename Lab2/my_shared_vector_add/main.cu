#include <stdio.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <chrono>

__global__
void initWith(float num, float* a, int N)
{

	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < N; i += stride)
	{
		a[i] = num;
	}
}

__global__
void addVectorsInto(float* result, float* a, float* b, int N)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < N; i += stride)
	{
		result[i] = a[i] + b[i];
	}
}

__global__
void addVectorsInto_cached(float* result, float* a, float* b, int N)
{
	extern __shared__ float shared_a[];
	float* shared_b = shared_a + blockDim.x;

	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	int tid = threadIdx.x;

	for (int i = index; i < N; i += stride)
	{
		shared_a[tid] = a[i];
		shared_b[tid] = b[i];

		result[i] = shared_a[tid] + shared_b[tid];

	}
}

__global__
void addVectorsInto_cached_sync(float* result, float* a, float* b, int N)
{
	extern __shared__ float shared_a[];
	float* shared_b = shared_a + blockDim.x;

	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	int tid = threadIdx.x;

	for (int i = index; i < N; i += stride)
	{
		shared_a[tid] = a[i];

		__syncthreads();

		shared_b[tid] = b[i];

		__syncthreads();

		result[i] = shared_a[tid] + shared_b[tid];

		__syncthreads();
	}
}

__global__
void addVectorsInto_cached_sync_static(float* result, float* a, float* b, int N)
{
	__shared__ float shared_a[256];
	__shared__ float shared_b[256];

	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	int tid = threadIdx.x;

	for (int i = index; i < N; i += stride)
	{
		shared_a[tid] = a[i];

		__syncthreads();

		shared_b[tid] = b[i];

		__syncthreads();

		result[i] = shared_a[tid] + shared_b[tid];

		__syncthreads();
	}
}

__global__
void addVectorsInto_bigcache(float* result, float* a, float* b, int N, int cacheSize)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	int tcs = cacheSize / blockDim.x;
	int cst = tcs * threadIdx.x;

	extern __shared__ float shared_res[];
	float *my_shared_res = shared_res + cst;

	int st_index = index;

	for (int i = index, ci = 0; i < N; i += stride, ++ci){
		my_shared_res[ci] = a[i];

		if (ci == tcs - 1) { // Місце в кещі закінчилось

			//__syncthreads();

			for (int i = st_index, ci = 0; ci < tcs; i += stride, ++ci)
				my_shared_res[ci] += b[i];

			//__syncthreads();

			for (int i = st_index, ci = 0; ci < tcs; i += stride, ++ci)
				result[i] = my_shared_res[ci];

			//__syncthreads();

			st_index = i + stride;
			ci = -1;
		}
	}

	//__syncthreads();

	for (int i = st_index, ci = 0; i < N; i += stride, ++ci) // last elements
		my_shared_res[ci] += b[i];

	//__syncthreads();

	for (int i = st_index, ci = 0; i < N; i += stride, ++ci)
		result[i] = my_shared_res[ci];

}


void checkElementsAre(float target, float* vector, int N)
{
	for (int i = 0; i < N; i++)
	{
		if (vector[i] != target)
		{
			printf("FAIL: vector[%d] - %0.0f does not equal %0.0f\n", i, vector[i], target);
			exit(1);
		}
	}
	printf("Success! All values calculated correctly.\n");
}

int main()
{
	int deviceId;
	int numberOfSMs;

	cudaGetDevice(&deviceId);
	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, deviceId);

	int maxShared = deviceProp.sharedMemPerBlockOptin;

	const int N = 2 << 24;
	size_t size = N * sizeof(float);

	float* c;

	c = (float*)malloc(size);

	float* da;
	float* db;
	float* dc;

	cudaMalloc(&da, size);
	cudaMalloc(&db, size);
	cudaMalloc(&dc, size);

	size_t threadsPerBlock;
	size_t numberOfBlocks;

	threadsPerBlock = 256;
	numberOfBlocks = 32 * numberOfSMs;

	cudaError_t addVectorsErr;
	cudaError_t asyncErr;

	initWith << <numberOfBlocks, threadsPerBlock >> > (3, da, N);
	initWith << <numberOfBlocks, threadsPerBlock >> > (4, db, N);
	initWith << <numberOfBlocks, threadsPerBlock >> > (0, dc, N);

	int cacheSize = maxShared / sizeof(float);

	const char* names[5] = { "addVectorsInto", "addVectorsInto_cached", "addVectorsInto_cached_sync", "addVectorsInto_cached_sync_static", "addVectorsInto_bigcache"};

	for(int t = 0; t < 3; ++t)
		for (int m = 0; m < 4; ++m)
			{
				auto start = std::chrono::steady_clock::now();

				
				for (int i = 0; i < 1000; ++i)
					if (m == 0)
						addVectorsInto << <numberOfBlocks, threadsPerBlock >> > (dc, da, db, N);
					else if (m == 1)
						addVectorsInto_cached << <numberOfBlocks, threadsPerBlock, threadsPerBlock * 2 * sizeof(float) >>> (dc, da, db, N);
					else if (m == 2)
						addVectorsInto_cached_sync << <numberOfBlocks, threadsPerBlock, threadsPerBlock * 2 * sizeof(float) >> > (dc, da, db, N);
					else if (m == 3)
						addVectorsInto_cached_sync_static << <numberOfBlocks, threadsPerBlock >> > (dc, da, db, N);
					else if (m == 4)
						addVectorsInto_bigcache << <numberOfBlocks, threadsPerBlock, 10 * sizeof(float) >> > (dc, da, db, N, 10);

				addVectorsErr = cudaGetLastError();
				if (addVectorsErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(addVectorsErr));
				asyncErr = cudaDeviceSynchronize();
				if (asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

				auto end = std::chrono::steady_clock::now();
				auto elapsed_ns = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
				printf("time = %d (%s)\n", (int)(elapsed_ns), names[m]);
			}


	cudaMemcpy(c, dc, size, cudaMemcpyDeviceToHost);

	checkElementsAre(7, c, N);

	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);

	free(c);
}
