#include <iostream>
#include <cmath>
#include <thread>
#include <vector>
#include <inttypes.h>

#include <SFML/System.hpp>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <imgui.h>
#include <imgui-SFML.h>
#include <implot.h>

#include "T1D.cuh"
#include "T2D.h"
#include "T2DImage.h"

using namespace std;

int main(int argc, char* argv[]) {
	T1D t1d;
	T2D t2d;
	T2DImage t2d_im;

	sf::RenderWindow window(sf::VideoMode(1600, 900), "Test Place");
	window.setFramerateLimit(60);
	ImGui::SFML::Init(window);
	ImPlot::CreateContext();

	sf::Clock deltaClock;
	sf::Clock Clock;
	while (window.isOpen()) {
		sf::Event event;
		while (window.pollEvent(event)) {
			ImGui::SFML::ProcessEvent(event);
			switch (event.type)
			{
			case sf::Event::Closed:
				window.close();
				break;
			case sf::Event::Resized:
				window.setView(sf::View(sf::FloatRect(0.f, 0.f, (float)event.size.width, (float)event.size.height)));
				break;
			}
		}

		ImGui::SFML::Update(window, deltaClock.restart());

		t1d.update();
		t2d.update();
		t2d_im.update();

		ImPlot::ShowDemoWindow();

		ImGui::SFML::Render(window);
		window.display();
		window.clear();
	}

	ImPlot::DestroyContext();
	ImGui::SFML::Shutdown();
}
