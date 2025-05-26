#include "T2DImage.h"

#include <imgui.h>
#include <imgui-SFML.h>
#include <SFML/Graphics/Sprite.hpp>

#include "TexReadout2D.cuh"


T2DImage::T2DImage() {
	in.loadFromFile("car1.png");
	in_tex.loadFromImage(in);
	d_in.update(in);

	updateN();
}

void T2DImage::updateN() {
	out_tex_p.setSize(N, N);
	out_tex_l.setSize(N, N);
	out_tex_pt.setSize(N, N);
	out_tex_lt.setSize(N, N);
}

void T2DImage::update() {
	if (ImGui::Begin("2D Image")) {
		if (ImGui::SliderInt("N", &N, 1, 1000))
			updateN();

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
			int scale = 200 / in_tex.getSize().x;

			ImGui::Text("Input texture, Size %d:%d, scale %d (%d:%d)", w, h, scale, w * scale, h * scale);

			sf::Sprite sp(in_tex);
			sp.setScale(scale, scale);
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
