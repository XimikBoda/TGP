#pragma once
#include <vector>
#include <SFML/System/Clock.hpp>
#include <SFML/Graphics/Image.hpp>
#include <cuda_runtime_api.h>
#include <SFML/Graphics/Texture.hpp>
using namespace std;

class Cuda2DTexture {
	cudaChannelFormatDesc channelDesc = 
		cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedNormalized8X4);
	cudaArray_t cuArray = NULL;
	struct cudaResourceDesc resDesc = {};
	struct cudaTextureDesc texDesc = {};

	int w = -1, h = -1;

	void release();
	void init(int w, int h);

public:
	cudaTextureObject_t texObj = 0;

	Cuda2DTexture() = default;

	Cuda2DTexture(sf::Image &im);

	void update(sf::Image &im);
};

class T2D {
	sf::Clock cl;
	float update_time = 1000;

	sf::Image in;
	sf::Texture in_tex;
	Cuda2DTexture d_in;

	int N = 100;
	sf::Texture out_tex;
	std::vector<uchar4> out;

	void ReadFetching2D();
public:
	T2D();

	void update();

};

