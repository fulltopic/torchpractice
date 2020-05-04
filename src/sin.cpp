/*
 * Notes:
 * 1. No batch_first(0), or the batch size would be input.size(1) = 1
 * 2. SGD optimizer is hard to converge
 * 3. state tensor declared but not defined, so that lstm would initialize them.
 * 4. state tensor is necessary for time sequence nn learning
 */

#include <torch/torch.h>
#include <ATen/NativeFunctions.h>


#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>

//#include "pytools/plotsin.h"
#include <matplotlibcpp.h>

using Tensor = torch::Tensor;

const float PI = M_PI;

void generateData(const std::string filePath, int exampleNum, int pointNum) {
	const int pointPerRad = 20;
	Tensor index = torch::range(0, exampleNum * pointNum - 1, 1).reshape({exampleNum, pointNum});
	std::cout << index.sizes() << std::endl;
	std::cout << index << std::endl;

	Tensor shift = torch::randint((-4 * pointPerRad), (4 * pointPerRad), {exampleNum, 1});
	std::cout << shift.sizes() << std::endl;
	std::cout << shift << std::endl;

	index = index.sub(shift).div(pointPerRad);
	std::cout << index << std::endl;

	auto data = torch::sin(index);
	std::cout << data << std::endl;

	torch::save(data, filePath);
}

Tensor readData(const std::string filePath) {
	Tensor dataTensor;
	torch::load(dataTensor, filePath);

//	std::cout << dataTensor.dim() << std::endl;
//	std::cout << "data " << std::endl;
//	std::cout << dataTensor << std::endl;
	return dataTensor;
}

std::vector<Tensor> createInput(Tensor dataTensor, float split) {
	const int trainNum = dataTensor.size(0) * split;
	const int validNum = dataTensor.size(0) - trainNum;
	const int pointNum = dataTensor.size(1);
	std::vector<Tensor> rc;

	Tensor trainTensor = torch::narrow(dataTensor, 0, 0, trainNum);
	Tensor inputTensor = torch::narrow(trainTensor, 1, 0, pointNum- 1);
	Tensor targetTensor = torch::narrow(trainTensor, 1, 1, pointNum - 1);
	rc.push_back(inputTensor);
	rc.push_back(targetTensor);

//	std::cout << "Input tensor " << std::endl;
//	std::cout << inputTensor << std::endl;
//	std::cout << "Target tensor " << std::endl;
//	std::cout << targetTensor << std::endl;

	Tensor validTensor = torch::narrow(dataTensor, 0, trainNum, validNum);
	Tensor validInputTensor = torch::narrow(validTensor, 1, 0, pointNum - 1);
	Tensor validTargetTensor = torch::narrow(validTensor, 1, 1, pointNum - 1);
	rc.push_back(validInputTensor);
	rc.push_back(validTargetTensor);

	return rc;
}


auto forwardStart = std::chrono::high_resolution_clock::now();
auto forwardStop = std::chrono::high_resolution_clock::now();
auto forwardSpan = forwardStop - forwardStart;

struct Net: torch::nn::Module {
	const int lstmHiddenSize;
//	at::native::CellParams param;
//	at::native::lstm_cell()
//	at::native::LSTMCell<at::native::CellParams> test;
	torch::nn::LSTM lstm0;
	torch::nn::LSTM lstm1;
	torch::nn::Linear fc;


	Net(int seqLen, int hiddenSize): lstmHiddenSize(hiddenSize),
			lstm0(torch::nn::LSTMOptions(seqLen, hiddenSize)),
			lstm1(torch::nn::LSTMOptions(hiddenSize, hiddenSize)),
			fc (hiddenSize, 1) {
		register_module("lstm0", lstm0);
		register_module("lstm1", lstm1);
		register_module("fc", fc);
	}

	Tensor forward (Tensor input) {
		forwardStart = std::chrono::high_resolution_clock::now();

		auto inputs = torch::chunk(input, input.size(1), 1);
		std::vector<Tensor> outputs;
//		Tensor state0 = torch::zeros({2, 1, input.size(0), lstmHiddenSize});
//		Tensor state1 = torch::zeros({2, 1, input.size(0), lstmHiddenSize});
		Tensor state00 = torch::zeros({1, input.size(0), lstmHiddenSize});
		Tensor state01 = torch::zeros({1, input.size(0), lstmHiddenSize});
		Tensor state10 = torch::zeros({1, input.size(0), lstmHiddenSize});
		Tensor state11 = torch::zeros({1, input.size(0), lstmHiddenSize});
		std::tuple<Tensor, Tensor> state0(state00, state01);
		std::tuple<Tensor, Tensor> state1(state10, state11);

		for (auto input: inputs) {
			input.set_requires_grad(true);
//			std::cout << "input size " << input.sizes() << std::endl;

			auto rnnOutput0 = lstm0->forward(input.view({1, input.size(0), input.size(1)}), state0);
			state0 = std::get<1>(rnnOutput0);
//			std::cout << "-----------------------------> End of lstm0 " << std::endl;
//			auto rnnOutput1 = lstm1->forward(rnnOutput0.output.view({input.size(0), lstmHiddenSize}), state1);
			auto rnnOutput1 = lstm1->forward(std::get<0>(rnnOutput0), state1);
			state1 = std::get<1>(rnnOutput1);
//			std::cout << "----------------------------> End of lstm1 " << std::endl;
			auto output = fc->forward(std::get<0>(rnnOutput1));
//			std::cout << "Output size: " << output.sizes() << std::endl;
//			auto output = fc->forward(rnnOutput0.output);

			outputs.push_back(output.view({output.size(1), output.size(2)}));
//			std::cout << "End of input " << std::endl;
		}
//		std::cout << "End of inputs " << std::endl;

		forwardStop = std::chrono::high_resolution_clock::now();
		forwardSpan += forwardStop - forwardStart;
//		std::cout << "End of forward" << std::endl;

		return torch::cat(outputs, 1);
	}
};


std::default_random_engine e;
void sinPlot(Tensor output, Tensor target, std::string fileName) {
	auto outputs = torch::chunk(output.detach(), output.size(0), 0);
	auto targets = torch::chunk(target, target.size(0), 0);

	int64_t index = e() % output.size(0);
//	int64_t index = 0;
//	std::cout << "Get index " << index << std::endl;

	const int64_t dataSize = output.size(1);
//	std::cout << "plot size: " << dataSize << std::endl;
	std::vector<int> xData(dataSize, 0);
	for (int i = 0; i < dataSize; i ++) {
		xData[i] = i;
	}

	std::vector<Tensor> y(2);
	y[0] = outputs[index];
	y[1] = targets[index];
//	y[0] = targets[index];
//	std::cout << "Output " << y[0] << std::endl;
	std::vector<std::string> colors(2);
	colors[0] = "r--";
	colors[1] = "g";

	float *outputDataPtr = y[0].data_ptr<float>();
	float *targetDataPtr = y[1].data_ptr<float>();
	std::vector<float> outputData(outputDataPtr, outputDataPtr + dataSize);
	std::vector<float> targetData(targetDataPtr, targetDataPtr + dataSize);

	matplotlibcpp::clf();
	matplotlibcpp::plot(outputData);
	matplotlibcpp::plot(targetData);
	matplotlibcpp::pause(1);
//	matplotlibcpp::draw();
//	plot(y, x, colors, fileName);
}

void train(const std::string filePath, float split) {
	const int SeqLen = 1;
	const int HiddenCell = 51;
	const int EpochNum = 12;
	const int BatchSize = 64;

	Tensor raw = readData(filePath);
	std::vector<Tensor> datas = createInput(raw, split);

	Tensor input = datas[0];
	Tensor target = datas[1];
	input.set_requires_grad(true);
	std::cout << "Target " << target.sizes() << std::endl;

	Net net(SeqLen, HiddenCell);
	auto pairs = net.named_parameters();
	for (auto pair: pairs) {
		std::cout << "param " << pair.key() << ": " << pair.value().sizes() << std::endl;
	}

	torch::optim::LBFGS optimizer (net.parameters(), torch::optim::LBFGSOptions(0.8));
//	std::string fileName = "./pipngs/sintest";


	int index = 0;
	auto cost = [&]() {
		optimizer.zero_grad();
		auto output = net.forward(input);
		sinPlot(output, target, "");
		auto loss = torch::mse_loss(output, target);
//		std::cout << "loss " << index << ": " << loss << std::endl;
		loss.backward();

		index ++;

		return loss;
	};

	net.train();
	for (int i = 0; i < EpochNum; i ++) {
		forwardSpan = forwardSpan.zero();

		std::cout << "Step " << i << std::endl;
		auto startIndex = index;
		auto start = std::chrono::high_resolution_clock::now();
		auto loss = optimizer.step(cost);
		auto stop = std::chrono::high_resolution_clock::now();
		std::cout << "loss " << i << loss << std::endl;
		std::cout << "run cost times: " << (index - startIndex)
				<< ", duration: " << std::chrono::duration_cast<std::chrono::seconds>(stop - start).count()
				<< "seconds" << std::endl;
		std::cout << "Forward occupies " << std::chrono::duration_cast<std::chrono::seconds>(forwardSpan).count() << "seconds " << std::endl;
	}

	{
		net.eval();
		Tensor validInput = datas[2];
		Tensor validTarget = datas[3];
		auto validOutput = net.forward(validInput);
		auto validLoss = torch::mse_loss(validOutput, validTarget);
		std::cout << "valid loss " << validLoss << std::endl;
		sinPlot(validOutput, validTarget, "");
	}
}

void testRead() {
	const std::string filePath = "./data/sin/testreaddata.pt";
	torch::Tensor saveTensor = torch::ones({10, 10});
	torch::save(saveTensor, filePath);

	torch::Tensor dataTensor = torch::zeros({10, 10});
	torch::load(dataTensor, filePath);
	std::cout << dataTensor.dim() << std::endl;
	for (int i = 0; i < dataTensor.dim(); i ++) {
		std::cout << dataTensor.size(i) << ", ";
	}
	std::cout << std::endl;

	auto data = dataTensor.accessor<float, 2>();
	for (int i = 0; i < data.size(0); i ++) {
		for (int j = 0; j < data.size(1); j ++) {
			std::cout << data[i][j] << ", ";
		}
		std::cout << std::endl;
	}
}

/*
 * t2.2xlarge performance after modification
 * Step 0
loss 00.504879
[ CPUFloatType{} ]
run cost times: 20, duration: 152seconds
Step 1
loss 10.00192627
[ CPUFloatType{} ]
run cost times: 16, duration: 121seconds
valid loss 0.00112198
[ CPUFloatType{} ]
 *
 */
int main() {
	matplotlibcpp::figure_size(1000, 1000);

//	testRead();
//	const std::string filePath = "./data/sin/testminidata.pt";
	const std::string filePath = "./data/sin/testdata1000.pt";
	const float split = 0.99;
//	generateData(filePath, 1000, 100);
//	auto tensor = readData(filePath);
//	createInput(tensor, 0.9);

	train(filePath, split);

}
