#pragma once
#include <vector>
#include <SFML/System/Clock.hpp>
using namespace std;

class T1D {
	sf::Clock cl;
	float update_time = 0.5;

	int in_N = 10;
	int out_N = 100;

	int in_max = 5;

	vector<float> in_x_0;
	vector<float> in_x_1;
	vector<float> in_y;

	vector<float> out_x;
	vector<float> out_x_tb;
	vector<float> out_y_0;
	vector<float> out_y_1;
	vector<float> out_y_2;
	vector<float> out_y_3;

	int addressMode = 0;
	const char* addressMode_names[4] = { "Wrap", "Clamp", "Mirror", "Border" };


	void update_in_size();
	void update_out_size();

public:
	T1D();

	void update();

};