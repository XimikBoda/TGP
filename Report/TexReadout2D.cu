#include "TexReadout2D.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void TexReadout2D(cudaTextureObject_t texObj, uchar4* out, int width, int height, float2 multiple, float2 offset)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		float u = multiple.x * x / (width - 1) + offset.x;
		float v = multiple.y * y / (height - 1) + offset.y;

		float4 c = tex2D<float4>(texObj, u, v);
		uchar4 uc = { 
			(uint8_t)(c.x * 255.f), 
			(uint8_t)(c.y * 255.f),
			(uint8_t)(c.z * 255.f),
			(uint8_t)(c.w * 255.f)
		};
		out[y * width + x] = uc;
	}
}

void cudaPrintIfError() {
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));
}

void ReadFetching2D(Cuda2DTexture &in, std::vector<uchar4> &out, int W, int H, bool tablelookup) {
	uchar4* d_out = 0;
	cudaMalloc(&d_out, sizeof(uchar4) * out.size()); // виділяємо пам'ять під виходні значення

	dim3 threadsPerBlock(16, 16);
	dim3 numberOfBlocks(
		(W + threadsPerBlock.x - 1) / threadsPerBlock.x,
		(H + threadsPerBlock.y - 1) / threadsPerBlock.y
	);

	float2 offset2 = { 0.f, 0.f };
	float2 multiple2 = { 1.f, 1.f };

	if (tablelookup) { // для реалізації Table Lookup
		offset2 = { 0.5f / (in.w), 0.5f / (in.h) };
		multiple2 = { (float)(in.w - 1) / in.w, (float)(in.h - 1) / in.h };
	}

	TexReadout2D<<<numberOfBlocks, threadsPerBlock>>>(in.texObj, d_out, W, H, multiple2, offset2); //запуск ядра

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));

	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));

	cudaMemcpy(out.data(), d_out, sizeof(uchar4) * out.size(), cudaMemcpyDeviceToHost); // копіювання реультату

	cudaFree(d_out);
}
