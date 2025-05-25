#pragma once
#include <SFML/System/Clock.hpp>
#include <SFML/Graphics/Image.hpp>
#include <SFML/Graphics/Texture.hpp>

#include "Cuda2DTexture.h"
#include "Pc2DTexture.h"

using namespace std;

class T2DImage {
	sf::Image in;
	sf::Texture in_tex;
	Cuda2DTexture d_in;

	int N = 100;
	Pc2DTexture out_tex_p;
	Pc2DTexture out_tex_l;
	Pc2DTexture out_tex_pt;
	Pc2DTexture out_tex_lt;

	int addressMode = 0;
	
	void updateN();
public:
	T2DImage();

	void update();

};

