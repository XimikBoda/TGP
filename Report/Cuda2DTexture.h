#pragma once
#include <SFML/Graphics/Image.hpp>
#include <SFML/System/NonCopyable.hpp>
#include <cuda_runtime_api.h>
using namespace std;

extern const char* addressMode_names[4];

class Cuda2DTexture : sf::NonCopyable {
	cudaChannelFormatDesc channelDesc = 
		cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedNormalized8X4);
	cudaArray_t cuArray = NULL;
	cudaResourceDesc resDesc = {};
	cudaTextureDesc texDesc = {};
	cudaTextureFilterMode filterMode = cudaFilterModePoint;
	cudaTextureAddressMode addressMode = cudaAddressModeWrap;

	void release();
	void init(int w, int h);

public:
	int w = -1, h = -1;
	bool tablelookup = false;

	cudaTextureObject_t texObj = 0;

	Cuda2DTexture() = default;

	Cuda2DTexture(sf::Image &im);

	~Cuda2DTexture();

	void update(sf::Image &im);
	void changeFM(cudaTextureFilterMode filterMode, bool tablelookup = false);
	void changeAM(cudaTextureAddressMode addressMode);
};