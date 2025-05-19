#include "T2D.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <imgui.h>
#include <imgui-SFML.h>
#include <SFML/Graphics/Sprite.hpp>

__global__ void TexReadout2D(cudaTextureObject_t texObj, uchar4* out, int width, int height, float2 multiple, float2 offset)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		float u = multiple.x * (float)x / (float)(width - 1) + offset.x;
		float v = multiple.y * (float)y / (float)(height - 1) + offset.y;

		float4 c = tex2D<float4>(texObj, u, v);
		uchar4 uc = { c.x * 255.f, c.y * 255.f, c.z * 255.f, c.w * 255.f };
		out[y * width + x] = uc;
	}
}

void cudaPrintIfError() {
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));
}

void Cuda2DTexture::release() {
	cudaDestroyTextureObject(texObj); // вивільнення ресурсів
	cudaFreeArray(cuArray);
}

void Cuda2DTexture::init(int w, int h) {
	cudaMallocArray(&cuArray, &channelDesc, w, h);

	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;

	texDesc.addressMode[0] = cudaAddressModeBorder;
	texDesc.addressMode[1] = cudaAddressModeBorder;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeNormalizedFloat;
	texDesc.normalizedCoords = true;

	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

	this->w = w, this->h = h;
}

void Cuda2DTexture::update(sf::Image& im) {
	if (w != im.getSize().x || h != im.getSize().y) {
		if (cuArray)
			release();
		init(im.getSize().x, im.getSize().y);
	}

	cudaMemcpy2DToArray(cuArray, 0, 0, im.getPixelsPtr(),
		w * 4, w * 4, h, cudaMemcpyHostToDevice);
}

Cuda2DTexture::Cuda2DTexture(sf::Image& im) {
	update(im);
}



T2D::T2D() {
	const sf::Color im_raw[4] =
	{
		sf::Color::Red, sf::Color::Green,
		sf::Color::Blue, sf::Color::Black,
	};

	in.create(2, 2, (sf::Uint8*)im_raw);
	in_tex.loadFromImage(in);
	d_in.update(in);
}

void T2D::ReadFetching2D() {
	out.resize(N * N);

	uchar4* d_out = 0;
	cudaMalloc(&d_out, sizeof(uchar4) * out.size()); // виділяємо пам'ять під виходні значення

	dim3 threadsPerBlock(16, 16);
	dim3 numberOfBlocks(
		(N + threadsPerBlock.x - 1) / threadsPerBlock.x,
		(N + threadsPerBlock.y - 1) / threadsPerBlock.y
	);

	bool tablelookup = false;

	float offset = tablelookup ? 1.f / (in.getSize().x) / 2.f : 0.f; // для реалізації Table Lookup
	float multiple = tablelookup ? (float)(in.getSize().x - 1) / in.getSize().x : 1.f;

	float2 offset2 = { offset , offset };
	float2 multiple2 = { multiple , multiple };

	TexReadout2D << <numberOfBlocks, threadsPerBlock >> > (d_in.texObj, d_out, N, N, multiple2, offset2); //запуск ядра

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));

	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));

	cudaMemcpy(out.data(), d_out, sizeof(uchar4) * out.size(), cudaMemcpyDeviceToHost); // копіювання реультату

	cudaFree(d_out);
}

void T2D::update() {
	if (ImGui::Begin("2D")) {
		if (cl.getElapsedTime().asSeconds() > update_time) {
			cl.restart();

			in.setPixel(1, 1, sf::Color(rand() % 256, rand() % 256, rand() % 256));
			in_tex.loadFromImage(in);
			d_in.update(in);
		}

		T2D::ReadFetching2D();

		out_tex.create(N, N);
		out_tex.update((sf::Uint8*)out.data());

		ImGui::SliderInt("N", &N, 1, 1000);
		ImGui::SliderFloat("Update time (in s) (0-1)", &update_time, 0, 1);
		ImGui::SliderFloat("Update time (in s) (1-60)", &update_time, 1, 60);

		ImGui::Text("Input texture");
		{
			sf::Sprite sp(in_tex);
			sp.setScale(100, 100);
			ImGui::Image(sp);
		}

		ImGui::Text("Output texture");
		{
			sf::Sprite sp(out_tex);
			int s = 200 / N;
			if (s < 1)
				s = 1;
			sp.setScale(s, s);
			ImGui::Image(sp);
		}


	}ImGui::End();;
}
