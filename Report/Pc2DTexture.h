#pragma once
#include <vector>
#include <cuda_runtime_api.h>
#include <SFML/Graphics/Texture.hpp>
using namespace std;

class Pc2DTexture : sf::NonCopyable {
	sf::Texture tex;
	vector<uchar4> raw_tex;
public:

	void setSize(int w, int h);
};