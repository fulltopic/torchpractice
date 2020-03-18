/*
 * lstmnet.cpp
 *
 *  Created on: Jan 22, 2020
 *      Author: zf
 */


#include <torch/torch.h>
#include "lmdbtools/LmdbReader.h"
#include "lmdbtools/LmdbDataDefs.h"
#include "nets/NetDef.h"
#include "nets/lstmnet.h"
//#include <matplotlibcpp.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <stdio.h>
#include <bits/stdc++.h>
#include <sys/types.h>
#include <filesystem>

using Tensor = torch::Tensor;
using TensorList = torch::TensorList;
using string = std::string;

bool lstmCompare(const Tensor& t0, const Tensor& t1) {
	return t0.size(0) > t1.size(0);
};

LstmNet::LstmNet(): conv0(torch::nn::Conv2dOptions(Conv0InChan, Conv0OutChan, {Conv0KernelH, Conv0KernelW}).stride({Conv0StrideH, Conv0StrideW})),
		conv1(torch::nn::Conv2dOptions(Conv1InChan, Conv1OutChan, {Conv1KernelH, Conv1KernelW})),
		lstm0(torch::nn::LSTM(torch::nn::LSTMOptions(GetLstm0Input(), Lstm0Hidden).batch_first(true))),
		fc(torch::nn::Linear(FcInput, FcOutput)),
		dataFile("./lstmdatafile.txt"),
		totalLen(0),
		totalSample(0)
{
	register_module("conv0", conv0);
	register_module("conv1", conv1);
	register_module("lstm0", lstm0);
	register_module("fc", fc);
}

LstmNet::~LstmNet() {
	dataFile.close();
}

Tensor LstmNet::forward (Tensor input) {
	Tensor lstmState;
	input.set_requires_grad(true);
	const int seqLen = input.size(0);
	const int batchSize = input.size(0) / seqLen;

	input = input.view({input.size(0), InputC, input.size(1), input.size(2)});
	std::cout << "input " << input.sizes() << std::endl;
	auto conv0Output = conv0->forward(input);
	std::cout << "conv0 output " << conv0Output.sizes() << std::endl;
	conv0Output = torch::relu(torch::max_pool2d(conv0Output, Conv0PoolSize));
	std::cout << "conv0 pool " << conv0Output.sizes() << std::endl;

	auto conv1Output = conv1->forward(conv0Output);
	std::cout << "conv1 output " << conv1Output.sizes() << std::endl;
	conv1Output = torch::relu(torch::max_pool2d(conv1Output, Conv1PoolSize));
	std::cout << "conv1 pool " << conv1Output.sizes() << std::endl;

	auto lstmInput = conv1Output.view({batchSize, seqLen, conv1Output.size(1) * conv1Output.size(2) * conv1Output.size(3)});
	auto lstmOutput = lstm0->forward(lstmInput, lstmState);
	lstmState = lstmOutput.state;

	auto fcOutput = fc->forward(lstmOutput.output);
	fcOutput = torch::relu(fcOutput);
	std::cout << "fcoutput " << fcOutput.sizes() << std::endl;

	//TODO: Not a good output
	return torch::log_softmax(fcOutput, 1);
}

Tensor LstmNet::forward(std::vector<Tensor> inputs) {
//		inputs.push_back(torch::ones({10, 5, 72}));

	const int batchSize = inputs.size();
	std::cout << "batchSize " << batchSize << std::endl;

	int maxSeqLen = 0;
	for (auto it = inputs.begin(); it != inputs.end(); it ++) {
		if (maxSeqLen < it->size(0)) {
			maxSeqLen = it->size(0);
		}
	}
	std::cout << "maxSeqLen: " << maxSeqLen << std::endl;

	std::vector<Tensor> convOutputs;
	for (int i = 0; i < inputs.size(); i ++) {
		Tensor input = inputs[i];
		input = input.view({input.size(0), InputC, input.size(1), input.size(2)});
		auto conv0Output = conv0->forward(input);
		conv0Output = torch::relu(torch::max_pool2d(conv0Output, {Conv0PoolH, Conv0PoolW}, {}, {1, 0}));

		auto conv1Output = conv1->forward(conv0Output);
		conv1Output = torch::relu(torch::max_pool2d(conv1Output, Conv1PoolSize));

		//The shape is {seqLen, channel * 2d}
		convOutputs.push_back(
				conv1Output.view({conv1Output.size(0), conv1Output.size(1) * conv1Output.size(2) * conv1Output.size(3)}));
	}
	std::sort(convOutputs.begin(), convOutputs.end(), lstmCompare);
	std::cout << "After sort " << std::endl;

	std::vector<int64_t> lengthVec(convOutputs.size(), 0);
	for (int i = 0; i < convOutputs.size(); i ++) {
		lengthVec[i] = convOutputs[i].size(0);
		std::cout << "length " << lengthVec[i] << std::endl;

		convOutputs[i] = torch::constant_pad_nd(convOutputs[i], {0, 0, 0, (maxSeqLen - lengthVec[i])}, 0);
	}

	Tensor lstmInput = at::stack(convOutputs, 0);
	std::cout << lstmInput.sizes() << std::endl;
	std::cout << "lengths " << std::endl;
	std::cout << torch::tensor(lengthVec) << std::endl;

	Tensor lstmInputData;
	Tensor lstmInputBatch;
	std::tie(lstmInputData, lstmInputBatch) = at::_pack_padded_sequence(
			at::stack(convOutputs, 0), torch::tensor(lengthVec), true);
	std::cout << "lstmInputData: " << lstmInputData.sizes() << std::endl;
	std::cout << "lstmInputBatch: " << lstmInputBatch.sizes() << std::endl;
	std::cout << lstmInputBatch << std::endl;

	Tensor lstmState = torch::zeros({2, Lstm0Layer * Lstm0Dir, (long)(lengthVec.size()), Lstm0Hidden});
	std::cout << "State sizes: " << lstmState.sizes() << std::endl;
	lstm0->enablePacked(lstmInputBatch);
	auto lstmRnnOutput = lstm0->forward(lstmInputData, lstmState);
	lstmState = lstmRnnOutput.state;
	auto lstmOutput = lstmRnnOutput.output;
	std::cout << "lstmOutput " << lstmOutput.sizes() << std::endl;

//		Tensor test0;
//		Tensor test1;
//		std::tie(test0, test1) = at::_pad_packed_sequence(lstmOutput, lstmInputBatch, true, 0, 0);
//		std::cout << "test0 " << test0.sizes() << std::endl;
//		std::cout << "test1 " << test1.sizes() << std::endl;
//		std::cout << test1 << std::endl;

	Tensor fcOutput = fc->forward(lstmOutput);
	std::cout << "fcOutput " << fcOutput.sizes() << std::endl;

	Tensor output = torch::log_softmax(fcOutput, OutputMax);

	return output;
}

Tensor LstmNet::forward(std::vector<Tensor> inputs, bool testLen) {
	for (int i = 0; i < inputs.size(); i ++) {
		totalLen += inputs[i].size(0);
		totalSample ++;
	}

	return inputs[0];
}

Tensor LstmNet::forward(std::vector<Tensor> inputs, const int SeqLen) {
	std::vector<Tensor> convOutputs;

	for (int i = 0; i < inputs.size(); i ++) {
		Tensor input = inputs[i];

//			if (input.size(0) < SeqLen) {
//				continue;
//			} else if (input.size(0) > SeqLen) {
//				input = input.narrow(0, input.size(0) - SeqLen, SeqLen);
//			}

		std::cout << "Narrowed input " << input.sizes() << std::endl;

		input = input.view({input.size(0), InputC, input.size(1), input.size(2)});
		auto conv0Output = conv0->forward(input);
		conv0Output = torch::relu(torch::max_pool2d(conv0Output, {Conv0PoolH, Conv0PoolW}, {}, {Conv0PoolPadH, Conv0PoolPadW}));

		auto conv1Output = conv1->forward(conv0Output);
		conv1Output = torch::relu(torch::max_pool2d(conv1Output, Conv1PoolSize));

		//The shape is {seqLen, channel * 2d}
		convOutputs.push_back(
				conv1Output.view({conv1Output.size(0), conv1Output.size(1) * conv1Output.size(2) * conv1Output.size(3)}));
	}
//		std::sort(convOutputs.begin(), convOutputs.end(), compare);
//		std::cout << "After sort " << std::endl;
//
//		std::vector<int64_t> lengthVec(convOutputs.size(), 0);
//		for (int i = 0; i < convOutputs.size(); i ++) {
//			lengthVec[i] = convOutputs[i].size(0);
//			std::cout << "length " << lengthVec[i] << std::endl;
//
//			convOutputs[i] = torch::constant_pad_nd(convOutputs[i], {0, 0, 0, (maxSeqLen - lengthVec[i])}, 0);
//		}

	Tensor lstmInput = at::stack(convOutputs, 0);
	std::cout << "lstminput " << lstmInput.sizes() << std::endl;
//		std::cout << "lengths " << std::endl;
//		std::cout << torch::tensor(lengthVec) << std::endl;

//		Tensor lstmInputData;
//		Tensor lstmInputBatch;
//		std::tie(lstmInputData, lstmInputBatch) = at::_pack_padded_sequence(
//				at::stack(convOutputs, 0), torch::tensor(lengthVec), true);
//		std::cout << "lstmInputData: " << lstmInputData.sizes() << std::endl;
//		std::cout << "lstmInputBatch: " << lstmInputBatch.sizes() << std::endl;
//		std::cout << lstmInputBatch << std::endl;

	Tensor lstmState = torch::zeros({2, Lstm0Layer * Lstm0Dir, (long)(convOutputs.size()), Lstm0Hidden});
	std::cout << "State sizes: " << lstmState.sizes() << std::endl;
//		lstm0->enablePacked(lstmInputBatch);
	auto lstmRnnOutput = lstm0->forward(lstmInput, lstmState);
	lstmState = lstmRnnOutput.state;
	auto lstmOutput = lstmRnnOutput.output;
	std::cout << "lstmOutput " << lstmOutput.sizes() << std::endl;

//		Tensor test0;
//		Tensor test1;
//		std::tie(test0, test1) = at::_pad_packed_sequence(lstmOutput, lstmInputBatch, true, 0, 0);
//		std::cout << "test0 " << test0.sizes() << std::endl;
//		std::cout << "test1 " << test1.sizes() << std::endl;
//		std::cout << test1 << std::endl;

	Tensor fcOutput = fc->forward(lstmOutput);
	std::cout << "fcOutput " << fcOutput.sizes() << std::endl;

	Tensor output = torch::log_softmax(fcOutput, OutputMax);

	return output.view({output.size(0) * output.size(1), output.size(2)});
}


Tensor LstmNet::forward(std::vector<Tensor> inputs, const int seqLen, bool isTrain, bool toRecord) {
	conv0->train(isTrain);
	conv1->train(isTrain);
	lstm0->train(isTrain);
	fc->train(isTrain);
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
	conv0Output = torch::relu(torch::max_pool2d(conv0Output, {Conv0PoolH, Conv0PoolW}, {}, {Conv0PoolPadH, Conv0PoolPadW}));

	auto conv1Output = conv1->forward(conv0Output);
	if (toRecord) {
		dataFile << "Conv0ouput -------------------------------------------> " << std::endl;
		dataFile << conv1Output << std::endl;
	}
	conv1Output = torch::relu(torch::max_pool2d(conv1Output, Conv1PoolSize));
	if (toRecord) {
		dataFile << "After pool -------------------------------------------> " << std::endl;
		dataFile << conv1Output << std::endl;
		dataFile << "============================================> END ==================================> " << std::endl;
	}
	//		std::cout << "Conv output: " << conv1Output.sizes() << std::endl;

//		Tensor lstmInput = at::stack(convOutputs, 0);
	Tensor lstmInput = conv1Output.view(
			{conv1Output.size(0) / seqLen, seqLen, conv1Output.size(1) * conv1Output.size(2) * conv1Output.size(3)});
//		std::cout << "lstminput: " << lstmInput.sizes() << std::endl;

//		Tensor lstmState = torch::zeros({2, Lstm0Layer * Lstm0Dir, (long)(convOutputs.size()), Lstm0Hidden});
//		std::cout << "State sizes: " << lstmState.sizes() << std::endl;

	Tensor lstmState;
	auto lstmRnnOutput = lstm0->forward(lstmInput, lstmState);
	lstmState = lstmRnnOutput.state;
	auto lstmOutput = lstmRnnOutput.output;
//		std::cout << "lstmOutput " << lstmOutput.sizes() << std::endl;

	Tensor fcOutput = fc->forward(lstmOutput);
//	std::cout << "fcOutput " << fcOutput.sizes() << std::endl;

	Tensor output = torch::log_softmax(fcOutput, OutputMax);

	return output.view({output.size(0) * output.size(1), output.size(2)});

//		return inputs[0];
}

