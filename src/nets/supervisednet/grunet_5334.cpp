/*
 * grunet_5334.cpp
 *
 *  Created on: May 18, 2020
 *      Author: zf
 */




/*
 * grunet_2523.cpp
 *
 *  Created on: May 13, 2020
 *      Author: zf
 */

#include "nets/supervisednet/grunet_5334.h"
#include <torch/script.h>

using std::string;
using std::vector;
using torch::Tensor;

thread_local torch::Tensor GruNet_5334::state;
thread_local bool GruNet_5334::resetState = false;

GruNet_5334::GruNet_5334():
		gru0(torch::nn::GRUOptions(360, 1024).batch_first(true)),
		batchNorm0(1),
		fc(1024, 42)
//		logger(Logger::GetLogger())
{
	register_module("gru0", gru0);
	register_module("batchNorm0", batchNorm0);
	register_module("fc", fc);
}

Tensor GruNet_5334::inputPreprocess(Tensor input) {
	return input.div(4);
}

void GruNet_5334::loadParams(const string modelPath) {
	std::cout << "Load model " << modelPath << std::endl;

	torch::serialize::InputArchive inChive;
	inChive.load_from(modelPath);
	this->load(inChive);

//	CNNGRUTransfer net(seqLen, loadNet);


	torch::OrderedDict<std::string, torch::Tensor>  params = this->named_parameters(true);
	for (auto ite = params.begin(); ite != params.end(); ite ++) {
		std::cout << ite->key() << ": " << ite->value().sizes() << std::endl;
	}
}

Tensor GruNet_5334::forward(vector<Tensor> inputs) {
//	Tensor input = inputs[0];
//	Tensor state = inputs[1];
//
//	Tensor lstmInput = inputPreprocess(input);
//	if (input.dim() == 2) {
//		lstmInput = input.view({1, 1, input.size(0) * input.size(1)});
//	}
//
//	//TODO: size of state
//	//TODO: Maintain state value between steps
//	auto gru0Output = gru0->forward(lstmInput, state);
//	auto batchNorm0Input = std::get<0>(gru0Output);
//	state = std::get<1>(gru0Output);
//
//	std::vector<Tensor> batchNorm0InputView;
//	for (int i = 0; i < 10; i ++) {
//		Tensor batchTensor = batchNorm0Input.clone();
//		batchNorm0InputView.push_back(batchTensor);
//	}
//	Tensor fcInput = at::cat(batchNorm0InputView, 0);
//	fcInput = fcInput.view({fcInput.size(1), fcInput.size(0), fcInput.size(2)});
//	std::cout << "fcInput sizes: " << fcInput.sizes() << std::endl;
//
//	auto batchNorm0Output = batchNorm0->forward(fcInput);
//	auto fcOutput = fc->forward(batchNorm0Output);
//
//	return fcOutput;
	return forward(inputs, false);
}

static void printData(const Tensor tensor, const std::string desc) {
	auto dataPtr = tensor.accessor<float, 3>();
	std::cout << desc << ": " << std::endl;
	for (int i = 0; i < tensor.size(2); i ++) {
		std::cout << dataPtr[0][0][i] << ", ";
		if ((i + 1) % 8 == 0) {
			std::cout << std::endl;
		}
	}
	std::cout << std::endl;
}

Tensor GruNet_5334::forward(std::vector<Tensor> inputs, bool isTrain) {
	//input sizes = {batch, 1, 360}
	//state sizes = {1, batch, 1024}
	Tensor input = inputs[0];

	if (resetState) {
		state = torch::zeros({1, input.size(0), 1024});
		resetState = false;
	}
//	Tensor state = inputs[1];

	//Does gru modify state ?
	auto gruOutput = gru0->forward(input, state);
	auto batchInput = std::get<0>(gruOutput);
	state = std::get<1>(gruOutput);

	auto batchOutput = batchNorm0->forward(batchInput);
	auto fcOutput = fc->forward(batchOutput);
//	auto fcOutput = fc->forward(batchInput);
	Tensor output = torch::log_softmax(fcOutput, 2);

	printData(fcOutput, "fcOutput");
	printData(fcOutput.exp(), "fcOutput exp");
	printData(output, "output");
//	auto smOutput = output.softmax(2);

//	return output;
	return output.view({output.size(0) * output.size(1), output.size(2)});
}

void GruNet_5334::reset() {
	resetState = true;
}


const string GruNet_5334::GetName() {
	return "GRUStep_seq_1";
}

