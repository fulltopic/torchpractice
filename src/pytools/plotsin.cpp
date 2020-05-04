#include <matplotlibcpp.h>
#include <cmath>
#include <vector>
#include <torch/torch.h>
#include <iostream>
#include <string>

void plot(std::vector<torch::Tensor> datas, torch::Tensor xAxis,
		std::vector<std::string> colors, std::string fileName) {
//	matplotlibcpp::figure_size(1200, 780);

	auto xData = xAxis.flatten().data<float>();
	std::vector<float> x(xAxis.numel());
	for (int i = 0; i < x.size(); i ++) {
		x[i] = xData[i];
	}
	for (int i = 0; i < datas.size(); i ++) {
		auto dataPtr = datas[i].flatten().data<float>();
		std::vector<float> data(datas[i].numel());
//		std::cout << "The plot data " << std::endl;
//		for (int j = 0; j < data.size(); j ++) {
//			data[j] = dataPtr[j];
//			std::cout << data[j] << ", ";
//		}
//		std::cout << std::endl;
		matplotlibcpp::plot(x, data, colors[i]);
		matplotlibcpp::pause(1);
	}

//	matplotlibcpp::show();
//	matplotlibcpp::legend();
//	matplotlibcpp::save(fileName);
}

void test() {
	int n = 5000;
	std::vector<double> x(n), y(n), z(n), w(n, 2);

	for (int i = 0; i < n; i ++) {
		x.at(i) = i * i;
		y.at(i) = sin((double)2 * M_PI * i / 360);
		z.at(i) = log((double)i);
	}

	matplotlibcpp::figure_size(1200, 780);
	matplotlibcpp::plot(x, y);
	matplotlibcpp::plot(x, w, "r--");
	matplotlibcpp::named_plot("log(x)", x, z);
	matplotlibcpp::xlim((int)0, 1000 * 1000);
	matplotlibcpp::title("Sample figure");
	matplotlibcpp::legend();
	matplotlibcpp::save("./testgraph.png");
}


