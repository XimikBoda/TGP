#include "T1D.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <imgui.h>
#include <implot.h>


__global__ void TexReadout1D(cudaTextureObject_t texObj, float* out, float offset, size_t N)
{
	for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x)
		out[i] = tex1D<float>(texObj, (float)i/(float)N + offset);
}

void ReadFetching1D(vector<float> &in, vector<float> &out, cudaTextureFilterMode filterMode, bool tablelookup = false, int threadsPerBlock = 256, int numberOfBlocks = 32) {
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
	texDesc.addressMode[0] = cudaAddressModeClamp;  // режим адреси
	texDesc.filterMode = filterMode;				// режим "отримання" текстури
	texDesc.normalizedCoords = true;				// нормалізація координат

	cudaTextureObject_t texObj = 0;
	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL); // створюємо об'єкт текстури

	float offset = tablelookup ? 1.f / (in.size()) / 2.f : 0.f; // для реалізації Table Lookup

	TexReadout1D<<<numberOfBlocks, threadsPerBlock>>>(texObj, d_out, offset, out.size()); //запуск ядра

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));

	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));

	cudaMemcpy(out.data(), d_out, sizeof(float) * out.size(), cudaMemcpyDeviceToHost); // копіювання реультату

	cudaDestroyTextureObject(texObj); // вивільнення ресурсів
	cudaFree(d_out);
	cudaFreeArray(cuArray);
}

void T1D::update_out_size() {
	out_x.resize(N);
	out_y_0.resize(N);
	out_y_1.resize(N);
	out_y_2.resize(N);
	out_y_3.resize(N);

	for (int i = 0; i < out_x.size(); ++i)
		out_x[i] = ((float)i / (float)(out_x.size()));
}

T1D::T1D() {
	cudaGetDevice(&deviceId);
	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);


	in_x.resize(10);
	in_y.resize(10);
	for (int i = 0; i < in_x.size(); ++i)
		in_y[i] = rand() % 5, in_x[i] = ((float)i/(in_x.size()));

	update_out_size();

	threadsPerBlock = 256;
	numberOfBlocks = 32 * numberOfSMs;
}

void T1D::update() {
	if (cl.getElapsedTime().asSeconds() > update_time) {
		cl.restart();

		for (int i = 0; i < in_x.size() - 1; ++i)
			in_y[i] = in_y[i + 1];
		in_y[in_x.size() - 1] = rand() % 5;
	}

	ReadFetching1D(in_y, out_y_0, cudaFilterModePoint, false, threadsPerBlock, numberOfBlocks);
	ReadFetching1D(in_y, out_y_1, cudaFilterModePoint, true, threadsPerBlock, numberOfBlocks);
	ReadFetching1D(in_y, out_y_2, cudaFilterModeLinear, false, threadsPerBlock, numberOfBlocks);
	ReadFetching1D(in_y, out_y_3, cudaFilterModeLinear, true, threadsPerBlock, numberOfBlocks);

	if (ImGui::Begin("Example")) {

		if (ImPlot::BeginPlot("My Plot")) {
			ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle);
			ImPlot::PlotLine("In", in_x.data(), in_y.data(), in_x.size());
			ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle);
			ImPlot::PlotLine("Out p", out_x.data(), out_y_0.data(), out_x.size());
			ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle);
			ImPlot::PlotLine("Out pt", out_x.data(), out_y_1.data(), out_x.size());
			ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle);
			ImPlot::PlotLine("Out l", out_x.data(), out_y_2.data(), out_x.size());
			ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle);
			ImPlot::PlotLine("Out lt", out_x.data(), out_y_3.data(), out_x.size());
			ImPlot::EndPlot();
		}

		if (ImGui::SliderInt("N", &N, 0, 1000))
			update_out_size();
		ImGui::SliderFloat("Update time (in s) (0-1)", &update_time, 0, 1);
		ImGui::SliderFloat("Update time (in s) (1-60)", &update_time, 1, 60);

	}ImGui::End();;
}
