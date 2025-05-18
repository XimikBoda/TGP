#pragma once
#include <vector>
#include <cuda_runtime_api.h>
#include <SFML/System/Clock.hpp>
using namespace std;

class T1D {
	sf::Clock cl;
	float update_time = 0.5;

	int N = 100;

	vector<float> in_x;
	vector<float> in_y;


	vector<float> out_x;
	vector<float> out_y_0;
	vector<float> out_y_1;
	vector<float> out_y_2;
	vector<float> out_y_3;


	int deviceId;
	int numberOfSMs;
	size_t threadsPerBlock;
	size_t numberOfBlocks;

	void update_out_size();
public:

	T1D();

	void update();

};