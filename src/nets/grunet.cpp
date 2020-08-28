/*
 * grunet.cpp
 *
 *  Created on: Jan 22, 2020
 *      Author: zf
 */


#include <torch/torch.h>
#include "lmdbtools/LmdbReader.h"
#include "lmdbtools/LmdbDataDefs.h"
#include "nets/NetDef.h"
#include "nets/grunet.h"
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
#include <cmath>
#include <matplotlibcpp.h>

using Tensor = torch::Tensor;
using TensorList = torch::TensorList;
using string = std::string;

bool lstmCompare(const Tensor& t0, const Tensor& t1) {
	return t0.size(0) > t1.size(0);
};

GRUNet::GRUNet(int inSeqLen):
		gru0(torch::nn::GRUOptions(360, 1024).num_layers(2).batch_first(true)),
		batchNorm0(inSeqLen),
		fc(1024, FcOutput),
		dataFile("./lstmdatafile.txt"),
		seqLen(inSeqLen)
{
//	register_module("lstm0", gru0);
	register_module("gru0", gru0);
	register_module("batchNorm0", batchNorm0);
	register_module("fc", fc);

	initParams();
}

GRUNet::~GRUNet() {
	dataFile.close();
}

void GRUNet::initParams() {
	auto params = this->named_parameters(true);
	for (auto ite = params.begin(); ite != params.end(); ite ++) {
		std::cout << "Get key " << ite->key() << std::endl;
//		if ((ite->key().compare("gru0.bias_ih_l0") == 0) || (ite->key().compare("gru0.bias_hh_l0") == 0)) {
		if ((ite->key().find("gru") != std::string::npos) && (ite->key().find("bias") != std::string::npos)) {
			auto dataPtr = ite->value().data_ptr<float>();
			std::cout << "bias samples before: " << dataPtr[0] << ", " << dataPtr[100] << ", " << dataPtr[1000] << std::endl;
			std::cout << ite->value().sizes() << std::endl;

			auto chunks = ite->value().chunk(3, 0);
			chunks[0].fill_(-1);

			std::cout << "bias samples after: " << dataPtr[0] << ", " << dataPtr[100] << ", " << dataPtr[1000] << std::endl;
		}

		if ((ite->key().find("gru") != std::string::npos) && (ite->key().find("weight") != std::string::npos)) {
			auto sizes = ite->value().sizes();
			int dataSize = 1;
			for (int i = 0; i < sizes.size(); i ++) {
				dataSize *= sizes[i];
			}
			int chunkSize = dataSize / 3;

			auto chunks = ite->value().chunk(3, 0);
			auto dataPtr = ite->value().data_ptr<float>();
			std::cout << "weights samples before: " << dataPtr[0] << ", " << dataPtr[chunkSize]
				<< ", " << dataPtr[chunkSize * 2] << std::endl;


			for (int i = 0; i < 3; i ++) {
//				torch::randn_out(chunks[0], chunks[0].sizes());
//				auto dataTensor = torch::randn_like(chunks[0]);
				auto dataTensor = torch::randn(chunks[i].sizes());
				auto meanTensor = torch::mean(dataTensor);
				std::cout << "Mean " << meanTensor.item<float>() << std::endl;
				auto varTensor = torch::var(dataTensor);
				std::cout << "Var " << varTensor.item<float>() << std::endl;
				dataTensor = dataTensor.div(sqrt((double)chunkSize));
//				std::cout << "Data sizes " << dataTensor.sizes() << std::endl;
//
//				torch::Tensor hisTensor = torch::histc(dataTensor);
//				auto hisData = hisTensor.data_ptr<float>();
//				std::vector<int> histVec(hisData, hisData + 100);
//				matplotlibcpp::hist(histVec);
//				matplotlibcpp::show();

				auto rndPtr = dataTensor.data_ptr<float>();
				auto chunkPtr = chunks[i].data_ptr<float>();
				for (int j = 0; j < chunkSize; j ++) {
					chunkPtr[j] = rndPtr[j];
				}
			}

			std::cout << "weights samples after: " << dataPtr[0] << ", " << dataPtr[chunkSize]
				<< ", " << dataPtr[chunkSize * 2] << std::endl;
		}

		if ((ite->key().find("fc") != std::string::npos) && (ite->key().find("weight") != std::string::npos)) {
			auto dataTensor = torch::randn(ite->value().sizes());
			dataTensor = dataTensor.div(sqrt(ite->value().numel()));
			auto rndPtr = dataTensor.data_ptr<float>();
			auto dataPtr = ite->value().data_ptr<float>();

			for (int j = 0; j < ite->value().numel(); j ++) {
				dataPtr[j] = rndPtr[j];
			}
			std::cout << "Initialized fc weight " << std::endl;
		}
	}
}


Tensor GRUNet::forward(std::vector<Tensor> inputs, const int seqLen, bool isTrain, bool toRecord) {
//	gru0->pretty_print(std::cout);

	gru0->train(isTrain);
	batchNorm0->train(isTrain);
	fc->train(isTrain);
//		std::cout << "Start forward " << std::endl;
	std::vector<Tensor> convOutputs;
	std::vector<Tensor> inputView;

	for (int i = 0; i < inputs.size(); i ++) {
		Tensor input = inputs[i];
//		std::cout << "Input " << input.sizes() << std::endl;
//		for (int j = 0; j < inputs[i].size(0); j ++) {
//			std::cout << inputs[i][j] << std::endl;
//		}
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
	auto lstm0RnnOutput = gru0->forward(lstmInput);
	auto lstm0Output = std::get<0>(lstm0RnnOutput);

	auto batchNorm0Output = batchNorm0->forward(lstm0Output);
//	lstmState = lstmRnnOutput.state;
//	auto lstmOutput = lstmRnnOutput.output;
	auto lstmOutput = batchNorm0Output;
//		std::cout << "lstmOutput " << lstmOutput.sizes() << std::endl;

	Tensor fcOutput = fc->forward(lstmOutput);
//	std::cout << "fcOutput " << fcOutput.sizes() << std::endl;

	Tensor output = torch::log_softmax(fcOutput, 2);
//	std::cout << "output " << output.sizes() << std::endl;
//	std::cout << "" << output[0][0] << std::endl;
//	Tensor smOutput = torch::softmax(fcOutput, 2);
//	std::cout << "smOutput " << smOutput.sizes() << std::endl;
//	std::cout << "" << smOutput << std::endl;

	return output.view({output.size(0) * output.size(1), output.size(2)});

//		return inputs[0];
}


torch::Tensor GRUNet::inputPreprocess(torch::Tensor input) {
	return input.div(4);
}

const std::string GRUNet::GetName() {
	return "GRUNet";
}
