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

	in.create(2, 2, (sf::Uint8*)im_raw);

	in_tex.loadFromImage(in);
	d_in.update(in);

	updateN();
}

void T2D::updateN() {
	out_tex_p.setSize(N, N);
	out_tex_l.setSize(N, N);
	out_tex_pt.setSize(N, N);
	out_tex_lt.setSize(N, N);
}


void T2D::randomize() {
	for (int y = 0; y < in.getSize().y; ++y)
		for (int x = 0; x < in.getSize().x; ++x)
			in.setPixel(x, y, sf::Color(rand() % 256, rand() % 256, rand() % 256));
	in_tex.loadFromImage(in);
	d_in.update(in);
}

void T2D::update() {
	if (ImGui::Begin("2D")) {
		if (cl.getElapsedTime().asSeconds() > update_time) {
			cl.restart();
			randomize();
		}

		if (ImGui::SliderInt("in N", &inN, 1, 100)) {
			in.create(inN, inN);
			randomize();
		}

		if (ImGui::SliderInt("Out N", &N, 1, 100))
			updateN();

		ImGui::SliderFloat("Update time (in s) (0-1)", &update_time, 0, 1);
		ImGui::SliderFloat("Update time (in s) (1-60)", &update_time, 1, 60);
		if (ImGui::Combo("Addressing mode", &addressMode, addressMode_names, 4))
			d_in.changeAM((cudaTextureAddressMode)addressMode);

		d_in.changeFM(cudaFilterModePoint, false);
		ReadFetching2D(d_in, out_tex_p);
		d_in.changeFM(cudaFilterModeLinear, false);
		ReadFetching2D(d_in, out_tex_l);
		d_in.changeFM(cudaFilterModePoint, true);
		ReadFetching2D(d_in, out_tex_pt);
		d_in.changeFM(cudaFilterModeLinear, true);
		ReadFetching2D(d_in, out_tex_lt);

		{
			int w = in_tex.getSize().x, h = in_tex.getSize().y;
			int s = 200 / in_tex.getSize().x;
			if (s < 1)
				s = 1;

			ImGui::Text("Input texture, Size %d:%d, scale %d (%d:%d)", w, h, s, w * s, h * s);

			sf::Sprite sp(in_tex);
			sp.setScale(s, s);
			ImGui::Image(sp);
		}

		{
			int s = 200 / N;
			if (s < 1)
				s = 1;
			ImGui::Text("Output texture, Size %d:%d, scale %d (%d:%d)", N, N, s, N * s, N * s);

			ImGui::Text("Nearest-Point");
			ImGui::SameLine(s * N + 20);
			ImGui::Text("Linear Filtering");

			sf::Sprite sp(out_tex_p.getTex());
			sp.setScale(s, s);
			ImGui::Image(sp);

			ImGui::SameLine();

			sp.setTexture(out_tex_l.getTex());
			ImGui::Image(sp);

			ImGui::SameLine(); ImGui::Text("No Table Lookup");

			sp.setTexture(out_tex_pt.getTex());
			ImGui::Image(sp);

			ImGui::SameLine();

			sp.setTexture(out_tex_lt.getTex());
			ImGui::Image(sp);

			ImGui::SameLine(); ImGui::Text("Table Lookup");
		}


	}ImGui::End();;
}
