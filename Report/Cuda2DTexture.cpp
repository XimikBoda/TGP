#include "Cuda2DTexture.h"

void Cuda2DTexture::release() {
	cudaDestroyTextureObject(texObj); // вивільнення ресурсів
	cudaFreeArray(cuArray);

	texObj = 0;
	cuArray = 0;
}

void Cuda2DTexture::init(int w, int h) {
	if (cuArray)
		release();

	this->w = w, this->h = h;

	cudaMallocArray(&cuArray, &channelDesc, w, h);

	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;

	texDesc.addressMode[0] = cudaAddressModeBorder;
	texDesc.addressMode[1] = cudaAddressModeBorder;
	texDesc.filterMode = filterMode;
	texDesc.readMode = cudaReadModeNormalizedFloat;
	texDesc.normalizedCoords = true;

	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
}

void Cuda2DTexture::update(sf::Image& im) {
	if (w != im.getSize().x || h != im.getSize().y) 
		init(im.getSize().x, im.getSize().y);

	cudaMemcpy2DToArray(cuArray, 0, 0, im.getPixelsPtr(),
		w * 4, w * 4, h, cudaMemcpyHostToDevice);
}

void Cuda2DTexture::changeFM(cudaTextureFilterMode filterMode) {
	if (filterMode != this->filterMode) {
		this->filterMode = filterMode;

		cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
	}
}

Cuda2DTexture::Cuda2DTexture(sf::Image& im) {
	update(im);
}

Cuda2DTexture::~Cuda2DTexture() {
	release();
}