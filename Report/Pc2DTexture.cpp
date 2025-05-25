#include "Pc2DTexture.h"

void Pc2DTexture::setSize(int w, int h) {
	if (this->w != w || this->h != h) {
		this->w = w, this->h = h;

		if (d_raw_tex)
			cudaFree(d_raw_tex);

		raw_tex.resize(w * h);
		cudaMalloc((void**)&d_raw_tex, w * h * sizeof(uchar4));
		tex.create(w, h);
	}
}

int2 Pc2DTexture::getSize() {
	return { w, h };
}

uchar4* Pc2DTexture::getDevicePtr() {
	return d_raw_tex;
}

const sf::Texture &Pc2DTexture::getTex() {
	return tex;
}

void Pc2DTexture::update() {
	cudaMemcpy(raw_tex.data(), d_raw_tex, w * h * sizeof(uchar4), cudaMemcpyDeviceToHost);
	tex.update((sf::Uint8*)raw_tex.data());
}

Pc2DTexture::~Pc2DTexture() {
	cudaFree(d_raw_tex);
}
