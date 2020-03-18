/*
 * FixedFcNet.cpp
 *
 *  Created on: Feb 1, 2020
 *      Author: zf
 */


#include "nets/FixedFcNet.h"

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

FixedFcNet::FixedFcNet(): def(LayerNum, InitChan),
//						inputBatch(InputC),
						conv0(torch::nn::Conv2dOptions(def.convChanns[0], def.convChanns[1], {def.convKernelHs[0], def.convKernelWs[0]})),
						batchNorm0(def.convChanns[1]),
						conv1(torch::nn::Conv2dOptions(def.convChanns[1], def.convChanns[2], {def.convKernelHs[1], def.convKernelWs[1]})),
						batchNorm1(def.convChanns[2]),
						conv2(torch::nn::Conv2dOptions(def.convChanns[2], def.convChanns[3], {def.convKernelHs[2], def.convKernelWs[2]})),
						batchNorm2(def.convChanns[3]),
						conv3(torch::nn::Conv2dOptions(def.convChanns[3], def.convChanns[4], {def.convKernelHs[3], def.convKernelWs[3]})),
						batchNorm3(def.convChanns[4]),
//						conv4(torch::nn::Conv2dOptions(def.convChanns[4], def.convChanns[5], {def.convKernelHs[4], def.convKernelWs[4]})),
//						batchNorm4(def.convChanns[5]),
						fc(torch::nn::Linear(def.getFcInput(), FcOutput)),
						dataFile("./fixedfcdatafile.txt")
{
//	register_module("inputBatch", inputBatch);
	register_module("conv0", conv0);
	register_module("batchNorm0", batchNorm0);
	register_module("conv1", conv1);
	register_module("batchNorm1", batchNorm1);
	register_module("conv2", conv2);
	register_module("batchNorm2", batchNorm2);
	register_module("conv3", conv3);
	register_module("batchNorm3", batchNorm3);
//	register_module("conv4", conv4);
//	register_module("batchNorm4", batchNorm4);
//	register_module("conv3_drop", conv3_drop);
	register_module("fc", fc);



	std::cout << "Conv0: " << conv0->weight.sizes() << std::endl;
	std::cout << "batchNorm0: " << batchNorm0->weight.sizes() << std::endl;
	std::cout << "conv1: " << conv1->weight.sizes() << std::endl;
	std::cout << "batchNorm1: " << batchNorm1->weight.sizes() << std::endl;
	std::cout << "conv2: " << conv2->weight.sizes() << std::endl;
	std::cout << "batchNorm2: " << batchNorm2->weight.sizes() << std::endl;
	std::cout << "conv3: " << conv3->weight.sizes() << std::endl;
	std::cout << "batchNorm3: " << batchNorm3->weight.sizes() << std::endl;
	std::cout << "Fc " << fc->weight.sizes() << std::endl;
//	std::cout << "Conv1: " << conv1->weight.sizes() << std::endl;
//	std::cout << "Conv2: " << conv2->weight.sizes() << std::endl;
	std::cout << "End of FixedFcNet constructor " << std::endl;
}

FixedFcNet::~FixedFcNet() {
	dataFile.close();
}

torch::Tensor FixedFcNet::inputPreprocess(torch::Tensor input) {
	return input.div(4);
}

torch::Tensor FixedFcNet::forward(std::vector<torch::Tensor> inputs, const int seqLen, bool isTrain, bool toRecord) {
//	inputBatch->train(isTrain);
	conv0->train(isTrain);
	batchNorm0->train(isTrain);
	conv1->train(isTrain);
	batchNorm1->train(isTrain);
	conv2->train(isTrain);
	batchNorm2->train(isTrain);
	conv3->train(isTrain);
//	conv3_drop->train(isTrain);
	batchNorm3->train(isTrain);
//	conv4->train(isTrain);
//	batchNorm4->train(isTrain);
	fc->train(isTrain);

	std::vector<Tensor> convOutputs;
	std::vector<Tensor> inputView;

	for (int i = 0; i < inputs.size(); i ++) {
		Tensor input = inputs[i];
//			std::cout << "Input " << input.sizes() << std::endl;
		inputView.push_back(input.view({input.size(0), InputC, input.size(1), input.size(2)}));
	}
//		std::cout << "End of view" << std::endl;

	Tensor conv0Input = at::cat(inputView, 0);
	conv0Input = inputPreprocess(conv0Input);
//		std::cout << "conv0Input " << conv0Input.sizes() << std::endl;

//	conv0Input = inputBatch->forward(conv0Input);
	if (toRecord) {
		dataFile << "Conv0Input: " << std::endl << conv0Input << std::endl;
	}

	auto conv0Output = conv0->forward(conv0Input);
	if (toRecord) {
//		dataFile << "Input " << std::endl << inputView[0] << std::endl;
		dataFile << "Weight0 " << std::endl << conv0->weight << std::endl;
		dataFile << "Conv0ouput -------------------------------------------> " << std::endl;
		dataFile << conv0Output[0] << std::endl;
	}
	conv0Output = torch::leaky_relu(torch::max_pool2d(conv0Output, {DefaultConvPoolH, ConvPoolW}));
	conv0Output = batchNorm0->forward(conv0Output);
//	std::cout << "End of conv0" << std::endl;

	auto conv1Output = conv1->forward(conv0Output);
	conv1Output = torch::leaky_relu(torch::max_pool2d(conv1Output, {DefaultConvPoolH, ConvPoolW}));
	conv1Output = batchNorm1->forward(conv1Output);
//	std::cout << "End of conv1 " << std::endl;


	auto conv2Output = conv2->forward(conv1Output);
	conv2Output = torch::leaky_relu(torch::max_pool2d(conv2Output, {DefaultConvPoolH, ConvPoolW}));
	conv2Output = batchNorm2->forward(conv2Output);

	auto conv3Output = conv3->forward(conv2Output);
	conv3Output = torch::leaky_relu(torch::max_pool2d(conv3Output, {DefaultConvPoolH, ConvPoolW}));
//	conv3Output = torch::dropout(conv3Output, 0.5, isTrain);
	conv3Output = batchNorm3->forward(conv3Output);

//	auto conv4Output = conv4->forward(conv3Output);
//	conv4Output = torch::leaky_relu(torch::max_pool2d(conv4Output, {DefaultConvPoolH, ConvPoolW}));
//	conv4Output = batchNorm4->forward(conv4Output);
//	std::cout << "End of conv2 " << std::endl;
//	if (toRecord) {
//		dataFile << "After pool -------------------------------------------> " << std::endl;
//		dataFile << conv0Output[0] << std::endl;
//		dataFile << "============================================> END ==================================> " << std::endl;
//	}
//			std::cout << "Conv output: " << conv1Output.sizes() << std::endl;

//		Tensor lstmInput = at::stack(convOutputs, 0);
//	Tensor lstmInput = conv1Output.view(
//			{conv1Output.size(0), conv1Output.size(1) * conv1Output.size(2) * conv1Output.size(3)});
	Tensor convOutput = conv3Output;
	Tensor lstmInput = convOutput.view(
			{convOutput.size(0), convOutput.size(1) * convOutput.size(2) * convOutput.size(3)});
//		std::cout << "lstminput: " << lstmInput.sizes() << std::endl;

//		Tensor lstmState = torch::zeros({2, Lstm0Layer * Lstm0Dir, (long)(convOutputs.size()), Lstm0Hidden});
//		std::cout << "State sizes: " << lstmState.sizes() << std::endl;

//	Tensor lstmState;
//	auto lstmRnnOutput = lstm0->forward(lstmInput, lstmState);
//	lstmState = lstmRnnOutput.state;
//	auto lstmOutput = lstmRnnOutput.output;
//		std::cout << "lstmOutput " << lstmOutput.sizes() << std::endl;

	Tensor fcOutput = fc->forward(lstmInput);
//	std::cout << "End of fc " << std::endl;
//		std::cout << "fcOutput " << fcOutput.sizes() << std::endl;

	Tensor output = torch::log_softmax(fcOutput, OutputMax);

	return output;

}

const std::string FixedFcNet::GetName() {
	return "FixedFcNet";
}
