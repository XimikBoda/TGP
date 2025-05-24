#pragma once
#include <SFML/System/Clock.hpp>
#include <SFML/Graphics/Image.hpp>
#include <SFML/Graphics/Texture.hpp>

#include "Cuda2DTexture.h"
#include "Pc2DTexture.h"

using namespace std;

class T2D {
	sf::Clock cl;
	float update_time = 1000;

	sf::Image in;
	sf::Texture in_tex;
	Cuda2DTexture d_in;

	int N = 100;
	sf::Texture out_tex;
	std::vector<uchar4> out;
public:
	T2D();

	void update();

};

