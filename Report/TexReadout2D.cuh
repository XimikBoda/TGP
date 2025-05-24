#pragma once
#include "Cuda2DTexture.h"
#include "Pc2DTexture.h"

using namespace std;

void ReadFetching2D(Cuda2DTexture& in, std::vector<uchar4>& out, int W, int H, bool tablelookup = false);
