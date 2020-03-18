#include "pytools/plotsin.h"

int main(){
	auto d1 = torch::ones({2, 2});
	auto d2 = torch::zeros({2, 2});
	std::vector<torch::Tensor> datas;
	datas.push_back(d1);
	datas.push_back(d2);

	auto x = torch::range(0, 3, 1);

	std::vector<std::string> colors;
	colors.push_back("r");
	colors.push_back("g");

	plot(datas, x, colors, "");
}
