/*
 * purefcnet.cpp
 *
 *  Created on: Feb 16, 2020
 *      Author: zf
 */


#include <torch/torch.h>
#include "lmdbtools/LmdbReader.h"
#include "lmdbtools/LmdbDataDefs.h"
#include "nets/NetDef.h"
#include "nets/purefcnet.h"
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

bool pureFcCompare(const Tensor& t0, const Tensor& t1) {
	return t0.size(0) > t1.size(0);
};

const int PureFcConvOutChan = 64;
const int ConvKernelH = 3;
const int ConvKernelW = 3;

int getFcInput() {
	return PureFcConvOutChan * (InputH - ConvKernelH + 1) * ((InputW - ConvKernelW + 1) / 2);
}

PureFcNet::PureFcNet():
		conv0(torch::nn::Conv2dOptions(1, PureFcConvOutChan, {ConvKernelH, ConvKernelW})),
		batchNorm0(PureFcConvOutChan),
		fc(torch::nn::Linear(getFcInput(), FcOutput)),
		dataFile("./purefcdatafile.txt")
{
	register_module("conv0", conv0);
	register_module("batchNorm0", batchNorm0);
	register_module("fc", fc);
}

PureFcNet::~PureFcNet() {
	dataFile.close();
}

Tensor PureFcNet::forward(std::vector<Tensor> inputs, const int seqLen, bool isTrain, bool toRecord) {
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


//	Tensor input = at::cat(inputs, 0);
//	input = input.view({input.size(0), input.size(1) * input.size(2)});

//	if (toRecord) {
//		dataFile << "Input " << std::endl << inputView[0] << std::endl;
//		dataFile << "Weight0 " << std::endl << conv0->weight << std::endl;
//		dataFile << "Conv0ouput -------------------------------------------> " << std::endl;
//		dataFile << conv0Output[0] << std::endl;
//	}

//	if (toRecord) {
//		dataFile << "After pool -------------------------------------------> " << std::endl;
//		dataFile << conv0Output[0] << std::endl;
//		dataFile << "============================================> END ==================================> " << std::endl;
//	}
//			std::cout << "Conv output: " << conv1Output.sizes() << std::endl;

//		Tensor lstmInput = at::stack(convOutputs, 0);
//		std::cout << "lstminput: " << lstmInput.sizes() << std::endl;

//		Tensor lstmState = torch::zeros({2, Lstm0Layer * Lstm0Dir, (long)(convOutputs.size()), Lstm0Hidden});
//		std::cout << "State sizes: " << lstmState.sizes() << std::endl;

//	Tensor lstmState;
//	auto lstmRnnOutput = lstm0->forward(lstmInput, lstmState);
//	lstmState = lstmRnnOutput.state;
//	auto lstmOutput = lstmRnnOutput.output;
//		std::cout << "lstmOutput " << lstmOutput.sizes() << std::endl;

	auto conv0Output = conv0->forward(conv0Input);
	conv0Output = torch::relu(torch::max_pool2d(conv0Output, {1, 2}));
	conv0Output = batchNorm0->forward(conv0Output);

	Tensor fcInput = conv0Output.view({conv0Output.size(0), conv0Output.size(1) * conv0Output.size(2) * conv0Output.size(3)});
	Tensor fcOutput = fc->forward(fcInput);
//	Tensor fcOutput = fc->forward(input);
//		std::cout << "fcOutput " << fcOutput.sizes() << std::endl;

	Tensor output = torch::log_softmax(fcOutput, OutputMax);

	return output;
//	return output.view({output.size(0) * output.size(1), output.size(2)});

//		return inputs[0];
}


