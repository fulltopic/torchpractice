/*
 * fcnet.cpp
 *
 *  Created on: Jan 22, 2020
 *      Author: zf
 */


#include <torch/torch.h>
#include "lmdbtools/LmdbReader.h"
#include "lmdbtools/LmdbDataDefs.h"
#include "nets/NetDef.h"
#include "nets/fcnet.h"
//#include <matplotlibcpp.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <stdio.h>
//#include <bits/stdc++.h>
//#include <sys/types.h>
#include <filesystem>

using Tensor = torch::Tensor;
using TensorList = torch::TensorList;
using string = std::string;

bool fcCompare(const Tensor& t0, const Tensor& t1) {
	return t0.size(0) > t1.size(0);
};

FcNet::FcNet(): conv0(torch::nn::Conv2dOptions(Conv0InChan, Conv0OutChan, {Conv0KernelH, Conv0KernelW}).stride({Conv0StrideH, Conv0StrideW})),
		batchNorm0(Conv1InChan),
		conv1(torch::nn::Conv2dOptions(Conv1InChan, Conv1OutChan, {Conv1KernelH, Conv1KernelW})),
		batchNorm1(Conv1OutChan),
		fc(torch::nn::Linear(GetLstm0Input(), FcOutput)),
		dataFile("./fcdatafile.txt")
{
	register_module("conv0", conv0);
	register_module("conv1", conv1);
	register_module("fc", fc);
}

FcNet::~FcNet() {
	dataFile.close();
}

Tensor FcNet::forward(std::vector<Tensor> inputs, const int seqLen, bool isTrain, bool toRecord) {
//	conv0->train(isTrain);
//	batchNorm0->train(isTrain);
//	conv1->train(isTrain);
//	batchNorm1->train(isTrain);
//	fc->train(isTrain);
//		std::cout << "Start forward " << std::endl;
	std::vector<Tensor> convOutputs;
	std::vector<Tensor> inputView;

	for (int i = 0; i < inputs.size(); i ++) {
		Tensor input = inputs[i];
//			std::cout << "Input " << input.sizes() << std::endl;
		inputView.push_back(input.view({input.size(0), InputC, input.size(1), input.size(2)}));
	}
//		std::cout << "End of view" << std::endl;

	Tensor conv0Input = at::cat(inputView, 0);
//		std::cout << "conv0Input " << conv0Input.sizes() << std::endl;

	auto conv0Output = conv0->forward(conv0Input);
	if (toRecord) {
		dataFile << "Input " << std::endl << inputView[0] << std::endl;
		dataFile << "Weight0 " << std::endl << conv0->weight << std::endl;
		dataFile << "Conv0ouput -------------------------------------------> " << std::endl;
		dataFile << conv0Output[0] << std::endl;
	}
	conv0Output = torch::leaky_relu(torch::max_pool2d(conv0Output, {Conv0PoolH, Conv0PoolW}, {}, {Conv0PoolPadH, Conv0PoolPadW}));
	conv0Output = batchNorm0->forward(conv0Output);

	auto conv1Output = conv1->forward(conv0Output);

	conv1Output = torch::leaky_relu(torch::max_pool2d(conv1Output, Conv1PoolSize));
	conv1Output = batchNorm1->forward(conv1Output);
//	if (toRecord) {
//		dataFile << "After pool -------------------------------------------> " << std::endl;
//		dataFile << conv0Output[0] << std::endl;
//		dataFile << "============================================> END ==================================> " << std::endl;
//	}
//			std::cout << "Conv output: " << conv1Output.sizes() << std::endl;

//		Tensor lstmInput = at::stack(convOutputs, 0);
	Tensor lstmInput = conv1Output.view(
			{conv1Output.size(0), conv1Output.size(1) * conv1Output.size(2) * conv1Output.size(3)});
//		std::cout << "lstminput: " << lstmInput.sizes() << std::endl;

//		Tensor lstmState = torch::zeros({2, Lstm0Layer * Lstm0Dir, (long)(convOutputs.size()), Lstm0Hidden});
//		std::cout << "State sizes: " << lstmState.sizes() << std::endl;

//	Tensor lstmState;
//	auto lstmRnnOutput = lstm0->forward(lstmInput, lstmState);
//	lstmState = lstmRnnOutput.state;
//	auto lstmOutput = lstmRnnOutput.output;
//		std::cout << "lstmOutput " << lstmOutput.sizes() << std::endl;

	Tensor fcOutput = fc->forward(lstmInput);
//		std::cout << "fcOutput " << fcOutput.sizes() << std::endl;

	Tensor output = torch::log_softmax(fcOutput, OutputMax);

	return output;
//	return output.view({output.size(0) * output.size(1), output.size(2)});

//		return inputs[0];
}
