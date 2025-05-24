#include "T2D.h"

#include <imgui.h>
#include <imgui-SFML.h>
#include <SFML/Graphics/Sprite.hpp>

#include "TexReadout2D.cuh"


T2D::T2D() {
	const sf::Color im_raw[4] =
	{
		sf::Color::Red, sf::Color::Green,
		sf::Color::Blue, sf::Color::Black,
	};

	//in.create(2, 2, (sf::Uint8*)im_raw);
	//in.create(10, 10);

	in.loadFromFile("car1.png");
	in_tex.loadFromImage(in);
	d_in.update(in);
}

void T2D::update() {
	if (ImGui::Begin("2D")) {
		if (cl.getElapsedTime().asSeconds() > update_time) {
			cl.restart();

			//in.setPixel(1, 1, sf::Color(rand() % 256, rand() % 256, rand() % 256));
			for (int y = 0; y < in.getSize().y; ++y)
				for (int x = 0; x < in.getSize().x; ++x)
					in.setPixel(x, y, sf::Color(rand() % 256, rand() % 256, rand() % 256));
			in_tex.loadFromImage(in);
			d_in.update(in);
		}


		out.resize(N * N);
		ReadFetching2D(d_in, out, N, N, false);

		out_tex.create(N, N);
		out_tex.update((sf::Uint8*)out.data());

		ImGui::SliderInt("N", &N, 1, 1000);
		ImGui::SliderFloat("Update time (in s) (0-1)", &update_time, 0, 1);
		ImGui::SliderFloat("Update time (in s) (1-60)", &update_time, 1, 60);

		ImGui::Text("Input texture");
		{
			sf::Sprite sp(in_tex);
			sp.setScale(200/in_tex.getSize().x, 200 / in_tex.getSize().x);
			ImGui::Image(sp);
		}

		ImGui::Text("Output texture");
		{
			sf::Sprite sp(out_tex);
			int s = 200 / N;
			if (s < 1)
				s = 1;
			sp.setScale(s, s);
			ImGui::Image(sp);
		}


	}ImGui::End();;
}
