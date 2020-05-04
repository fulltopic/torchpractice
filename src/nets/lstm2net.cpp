/*
 * lstm2net.cpp
 *
 *  Created on: Apr 3, 2020
 *      Author: zf
 */


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
#include "nets/lstm2net.h"
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

//static bool lstmCompare(const Tensor& t0, const Tensor& t1) {
//	return t0.size(0) > t1.size(0);
//};

//batchnorm registered
Lstm2Net::Lstm2Net(int inSeqLen):
		lstm0(torch::nn::LSTMOptions(360, 1024).batch_first(true)),
		batchNorm0(inSeqLen),
		lstm1(torch::nn::LSTMOptions(1024, 1024).batch_first(true)), //TODO: batchfirst?
		batchNorm1(inSeqLen),
		fc(1024, FcOutput),
		dataFile("./lstmdatafile.txt"),
		seqLen(inSeqLen)
{
	register_module("lstm0", lstm0);
	register_module("batchNorm0", batchNorm0);
	register_module("lstm1", lstm1);
	register_module("batchNorm1", batchNorm1);
	register_module("fc", fc);
}

Lstm2Net::~Lstm2Net() {
	dataFile.close();
}


//TODO
Tensor Lstm2Net::forward(std::vector<Tensor> inputs, const int seqLen, bool isTrain, bool toRecord) {
	lstm0->train(isTrain);
	fc->train(isTrain);
//		std::cout << "Start forward " << std::endl;
	std::vector<Tensor> convOutputs;
	std::vector<Tensor> inputView;

	for (int i = 0; i < inputs.size(); i ++) {
		Tensor input = inputs[i];
//			std::cout << "Input " << input.sizes() << std::endl;
//		inputView.push_back(input.view({input.size(0), InputC, input.size(1), input.size(2)}));
		inputView.push_back(input);
	}
//		std::cout << "End of view" << std::endl;

	Tensor rawInput = at::cat(inputView, 0);
	rawInput = inputPreprocess(rawInput);
//		std::cout << "conv0Input " << conv0Input.sizes() << std::endl;



//		Tensor lstmInput = at::stack(convOutputs, 0);
	Tensor lstmInput = rawInput.view(
			{rawInput.size(0) / seqLen, seqLen, rawInput.size(1) * rawInput.size(2)});
//		std::cout << "lstminput: " << lstmInput.sizes() << std::endl;

//		Tensor lstmState = torch::zeros({2, Lstm0Layer * Lstm0Dir, (long)(convOutputs.size()), Lstm0Hidden});
//		std::cout << "State sizes: " << lstmState.sizes() << std::endl;

//	Tensor lstmState;
	auto lstm0RnnOutput = lstm0->forward(lstmInput);
	auto lstm0Output = std::get<0>(lstm0RnnOutput);

	auto batchNorm0Output = batchNorm0->forward(lstm0Output);

	auto lstm1RnnOutput = lstm1->forward(batchNorm0Output);
	auto lstm1Output = std::get<0>(lstm1RnnOutput);
	auto batchNorm1Output = batchNorm1->forward(lstm1Output);
//	lstmState = lstmRnnOutput.state;
//	auto lstmOutput = lstmRnnOutput.output;
	auto lstmOutput = batchNorm1Output;
//		std::cout << "lstmOutput " << lstmOutput.sizes() << std::endl;

	Tensor fcOutput = fc->forward(lstmOutput);
//	std::cout << "fcOutput " << fcOutput.sizes() << std::endl;

	Tensor output = torch::log_softmax(fcOutput, OutputMax);

	return output.view({output.size(0) * output.size(1), output.size(2)});

//		return inputs[0];
}


torch::Tensor Lstm2Net::inputPreprocess(torch::Tensor input) {
	return input.div(4);
}

const std::string Lstm2Net::GetName() {
	return "Lstm2Net";
}


