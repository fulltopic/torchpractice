/*
 * cnnnet.cpp
 *
 *  Created on: Mar 16, 2020
 *      Author: zf
 */

#include "nets/cnnnet.h"

#include <torch/torch.h>
#include "lmdbtools/LmdbReader.h"
#include "lmdbtools/LmdbDataDefs.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <stdio.h>
#include <filesystem>

using Tensor = torch::Tensor;
using TensorList = torch::TensorList;
using string = std::string;

CNNNet::CNNNet() :
		conv0(torch::nn::Conv2dOptions(1, 32, {5, 3}).padding({2, 1})),
		batchNorm0(32),
		conv1(torch::nn::Conv2dOptions(32, 128, {3, 3}).padding({1, 1})),
		batchNorm1(128),
		fc0(torch::nn::Linear(23040, 1024)),
		fcBatchNorm0(1024),
		fc1(torch::nn::Linear(1024, 42)),
		dataFile("./cnnnet.txt")
{
	register_module("conv0", conv0);
	register_module("batchNorm0", batchNorm0);
	register_module("conv1", conv1);
	register_module("batchNorm1", batchNorm1);
	register_module("fc0", fc0);
	register_module("fcBatchNorm0", fcBatchNorm0);
	register_module("fc1", fc1);
}

CNNNet::~CNNNet() {
	dataFile.close();
}

torch::Tensor CNNNet::inputPreprocess(torch::Tensor input) {
	return input.div(4);
}

void CNNNet::setTrain(bool isTrain) {
	conv0->train(isTrain);
	batchNorm0->train(isTrain);
	fc0->train(isTrain);
	fcBatchNorm0->train(isTrain);
	fc1->train(isTrain);
}

torch::Tensor CNNNet::forward(std::vector<torch::Tensor> inputs, const int seqLen, bool isTrain, bool toRecord) {
	setTrain(isTrain);

	std::vector<Tensor> convOutputs;
	std::vector<Tensor> inputView;

	for (int i = 0; i < inputs.size(); i ++) {
		Tensor input = inputs[i];
//			std::cout << "Input " << input.sizes() << std::endl;
		inputView.push_back(input.view({input.size(0), 1, input.size(1), input.size(2)}));
	}

	Tensor conv0Input = at::cat(inputView, 0);
	conv0Input = inputPreprocess(conv0Input);

	auto conv0Output = conv0->forward(conv0Input);
	conv0Output = torch::max_pool2d(conv0Output, {1, 2});
	conv0Output = torch::leaky_relu(conv0Output);

	auto batch0Output = batchNorm0->forward(conv0Output);

	auto conv1Output = conv1->forward(batch0Output);
	conv1Output = torch::max_pool2d(conv1Output, {1, 2});
	conv1Output = torch::leaky_relu(conv1Output);

	auto batch1Output = batchNorm1->forward(conv1Output);

	Tensor convOutput = batch1Output;
//	std::cout << "Con0output " << convOutput.sizes() << std::endl;
	Tensor fcInput = convOutput.view(
			{convOutput.size(0), convOutput.size(1) * convOutput.size(2) * convOutput.size(3)});
	auto fc0Output = fc0->forward(fcInput);

	auto fc1Input = fcBatchNorm0->forward(fc0Output);

	auto fc1Output = fc1->forward(fc1Input);

	auto fcOutput = fc1Output;
	Tensor output = torch::log_softmax(fcOutput, 1);

	return output;
}

const std::string CNNNet::GetName() {
	return "CNNNet";
}
