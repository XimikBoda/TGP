#pragma once
#include <vector>
#include <cuda_runtime_api.h>
#include <SFML/Graphics/Texture.hpp>
using namespace std;

class Pc2DTexture : sf::NonCopyable {
	sf::Texture tex;
	vector<uchar4> raw_tex;
	int w = -1, h = -1;
	uchar4* d_raw_tex = 0;
public:

	void setSize(int w, int h);

	int2 getSize();
	uchar4* getDevicePtr();
	const sf::Texture &getTex();

	void update();

	~Pc2DTexture();
};