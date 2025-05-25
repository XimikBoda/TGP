#include "T1D.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <imgui.h>
#include <implot.h>


__global__ void TexReadout1D(cudaTextureObject_t texObj, float* out, float multiple, float offset, size_t out_N)
{
	for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < out_N; i += gridDim.x * blockDim.x)
		out[i] = tex1D<float>(texObj, multiple * i + offset);
}

void ReadFetching1D(vector<float> &in, vector<float> &out, cudaTextureFilterMode filterMode, cudaTextureAddressMode addressMode, bool tablelookup = false, int threadsPerBlock = 256) {
	cudaChannelFormatDesc channelDesc =
		cudaCreateChannelDesc(sizeof(float) * 8, 0, 0, 0, cudaChannelFormatKindFloat);

	cudaArray_t cuArray;
	cudaMallocArray(&cuArray, &channelDesc, in.size()); // створюмо масив вхідних значень
	cudaMemcpyToArray(cuArray, 0, 0, in.data(), in.size() * sizeof(float), cudaMemcpyHostToDevice); // копіюємо в нього

	float* d_out = 0;
	cudaMalloc(&d_out, sizeof(float) * out.size()); // виділяємо пам'ять під виходні значення

	struct cudaResourceDesc resDesc = {};			// налаштовуємо опис ресурсу
	resDesc.resType = cudaResourceTypeArray; 
	resDesc.res.array.array = cuArray;				// на створений масив

	struct cudaTextureDesc texDesc = {};			// налаштовуємо опис текстури
	texDesc.addressMode[0] = addressMode;  // режим адреси
	texDesc.filterMode = filterMode;				// режим "отримання" текстури
	texDesc.normalizedCoords = true;				// нормалізація координат

	cudaTextureObject_t texObj = 0;
	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL); // створюємо об'єкт текстури

	float offset = tablelookup ? 1.f / (in.size()) / 2.f : 0.f; // для реалізації Table Lookup
	float multiple = tablelookup 
		? (float)(in.size() - 1) / in.size() / (out.size() - 1) 
		: 1.f / out.size();

	size_t numberOfBlocks = (out.size() + threadsPerBlock - 1) / threadsPerBlock;

	TexReadout1D<<<numberOfBlocks, threadsPerBlock>>>(texObj, d_out, multiple, offset, out.size()); //запуск ядра

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));

	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));

	cudaMemcpy(out.data(), d_out, sizeof(float) * out.size(), cudaMemcpyDeviceToHost); // копіювання реультату

	cudaDestroyTextureObject(texObj); // вивільнення ресурсів
	cudaFree(d_out);
	cudaFreeArray(cuArray);
}

void T1D::update_in_size() {
	in_x_0.resize(in_N);
	in_x_1.resize(in_N);
	in_y.resize(in_N);
	for (int i = 0; i < in_x_0.size(); ++i)
		in_y[i] = rand() % (in_max + 1),
		in_x_0[i] = ((float)i / (in_x_0.size()) + 0.5f / in_x_0.size()),
		in_x_1[i] = ((float)i / (in_x_0.size() - 1));
}

void T1D::update_out_size() {
	out_x.resize(out_N);
	out_x_tb.resize(out_N);
	out_y_0.resize(out_N);
	out_y_1.resize(out_N);
	out_y_2.resize(out_N);
	out_y_3.resize(out_N);

	for (int i = 0; i < out_x.size(); ++i)
		out_x[i] = ((float)i / out_x.size()),
		out_x_tb[i] = ((float)i / (out_x.size() - 1));
}

T1D::T1D() {
	update_in_size();
	update_out_size();
}

void T1D::update() {
	if (cl.getElapsedTime().asSeconds() > update_time) {
		cl.restart();

		for (int i = 0; i < in_x_0.size() - 1; ++i)
			in_y[i] = in_y[i + 1];
		in_y[in_x_0.size() - 1] = rand() % (in_max + 1);
	}

	cudaTextureAddressMode am = (cudaTextureAddressMode)addressMode;

	size_t threadsPerBlock = 256;
	ReadFetching1D(in_y, out_y_0, cudaFilterModePoint, am, false, threadsPerBlock);
	ReadFetching1D(in_y, out_y_1, cudaFilterModePoint, am, true, threadsPerBlock);
	ReadFetching1D(in_y, out_y_2, cudaFilterModeLinear, am, false, threadsPerBlock);
	ReadFetching1D(in_y, out_y_3, cudaFilterModeLinear, am, true, threadsPerBlock);

	if (ImGui::Begin("1D")) {

		if (ImPlot::BeginPlot("Just")) {
			ImPlot::SetupAxes("X-Axis 1", "Y-Axis 1");
			ImPlot::SetupAxesLimits(0, 1, 0, in_max, ImPlotCond_Always);

			ImPlot::SetupAxis(ImAxis_X2, "X-Axis 2", ImPlotAxisFlags_AuxDefault);
			ImPlot::SetupAxisLimits(ImAxis_X2, 0, in_x_0.size(), ImPlotCond_Always);

			ImPlot::SetAxes(ImAxis_X2, ImAxis_Y1);
			ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.25f);
			ImPlot::PlotBars("In bars", in_y.data(), in_x_0.size(), 0.9, 0.5);
			ImPlot::PopStyleVar();

			ImPlot::SetAxes(ImAxis_X1, ImAxis_Y1);
			ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle);
			ImPlot::PlotLine("In line", in_x_0.data(), in_y.data(), in_x_0.size());
			ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle);
			ImPlot::PlotLine("Out point", out_x.data(), out_y_0.data(), out_x.size());
			ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle);
			ImPlot::PlotLine("Out line", out_x.data(), out_y_2.data(), out_x.size());
			ImPlot::EndPlot();
		}

		if (ImPlot::BeginPlot("Table Lookup")) {
			ImPlot::SetupAxes("X-Axis 1", "Y-Axis 1");
			ImPlot::SetupAxesLimits(0, 1, 0, in_max, ImPlotCond_Always);

			ImPlot::SetupAxis(ImAxis_X2, "X-Axis 2", ImPlotAxisFlags_AuxDefault);
			ImPlot::SetupAxisLimits(ImAxis_X2, 0, in_x_1.size() - 1, ImPlotCond_Always);

			ImPlot::SetAxes(ImAxis_X2, ImAxis_Y1);
			ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.25f);
			ImPlot::PlotBars("In bars", in_y.data(), in_x_1.size(), 0.9);
			ImPlot::PopStyleVar();

			ImPlot::SetAxes(ImAxis_X1, ImAxis_Y1);
			ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle);
			ImPlot::PlotLine("In line", in_x_1.data(), in_y.data(), in_x_1.size());
			ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle);
			ImPlot::PlotLine("Out point", out_x_tb.data(), out_y_1.data(), out_x.size());
			ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle);
			ImPlot::PlotLine("Out line", out_x_tb.data(), out_y_3.data(), out_x.size());
			ImPlot::EndPlot();
		}

		ImGui::Combo("Wave form", &addressMode, addressMode_names, 4);

		if (ImGui::SliderInt("In N", &in_N, 1, 1000))
			update_in_size();
		if (ImGui::SliderInt("Out N", &out_N, 1, 1000))
			update_out_size();

		ImGui::SliderInt("In max", &in_max, 0, 1000);

		ImGui::SliderFloat("Update time (in s) (0-1)", &update_time, 0, 1);
		ImGui::SliderFloat("Update time (in s) (1-60)", &update_time, 1, 60);

	}ImGui::End();;
}
