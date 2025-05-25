#include "TexReadout2D.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void TexReadout2D(cudaTextureObject_t texObj, uchar4* out, int2 size, float2 multiple, float2 offset)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < size.x && y < size.y) {
		float u = multiple.x * x + offset.x;
		float v = multiple.y * y + offset.y;

		float4 c = tex2D<float4>(texObj, u, v);
		uchar4 uc = { 
			(uint8_t)(c.x * 255.f), 
			(uint8_t)(c.y * 255.f),
			(uint8_t)(c.z * 255.f),
			(uint8_t)(c.w * 255.f)
		};
		out[y * size.x + x] = uc;
	}
}

void cudaPrintIfError() {
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));
}

void ReadFetching2D(Cuda2DTexture &in, Pc2DTexture &out) {
	int2 outSize = out.getSize();

	dim3 threadsPerBlock(16, 16);
	dim3 numberOfBlocks(
		(outSize.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
		(outSize.y + threadsPerBlock.y - 1) / threadsPerBlock.y
	);

	float2 offset2 = { 0.f, 0.f };
	float2 multiple2 = { 1.f / outSize.x, 1.f / outSize.y };

	if (in.tablelookup) { // для реалізації Table Lookup
		offset2 = { 0.5f / (in.w), 0.5f / (in.h) };
		multiple2 = { (float)(in.w - 1) / in.w / (outSize.x - 1), 
			(float)(in.h - 1) / in.h / (outSize.y - 1) };
	}

	multiple2.x *= 1;
	multiple2.y *= 1;

	TexReadout2D<<<numberOfBlocks, threadsPerBlock>>>(in.texObj, out.getDevicePtr(), outSize, multiple2, offset2); //запуск ядра

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));

	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));

	out.update();
}
