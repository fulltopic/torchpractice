/*
 * fcgrunet.cpp
 *
 *  Created on: Apr 2, 2020
 *      Author: zf
 */


#include "nets/fcgrunet.h"


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

//bool lstmCompare(const Tensor& t0, const Tensor& t1) {
//	return t0.size(0) > t1.size(0);
//};

FcGRUNet::FcGRUNet(int inSeqLen):
		fcPre(360, 1024),
		batchNormPre(1024),
		gru0(torch::nn::GRUOptions(1024, 1024).batch_first(true)),
		batchNorm0(inSeqLen),
		fc(1024, FcOutput),
		dataFile("./lstmdatafile.txt"),
		seqLen(inSeqLen)
{
//	register_module("lstm0", gru0);
	register_module("fcPre", fcPre);
	register_module("batchNormPre", batchNormPre);
	register_module("gru0", gru0);
	register_module("batchNorm0", batchNorm0);
	register_module("fc", fc);

	initParams();
}

FcGRUNet::~FcGRUNet() {
	dataFile.close();
}

void FcGRUNet::initParams() {
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
			std::cout << "Initialized fc weight " << ite->key() << std::endl;
		}
	}
}


Tensor FcGRUNet::forward(std::vector<Tensor> inputs, const int seqLen, bool isTrain, bool toRecord) {
//	gru0->pretty_print(std::cout);

	fcPre->train(isTrain);
	batchNormPre->train(isTrain);
	gru0->train(isTrain);
	batchNorm0->train(isTrain);
	fc->train(isTrain);

	std::vector<Tensor> convOutputs;
	std::vector<Tensor> inputView;


	for (int i = 0; i < inputs.size(); i ++) {
		Tensor input = inputs[i];
		inputView.push_back(input.view({input.size(0), input.size(1) * input.size(2)}));
	}

	Tensor rawInput = at::cat(inputView, 0);
	rawInput = inputPreprocess(rawInput);

	auto fcPreOutput = fcPre->forward(rawInput);
	fcPreOutput = torch::relu(fcPreOutput);
	auto normPreOutput = batchNormPre->forward(fcPreOutput);

	Tensor lstmInput = normPreOutput.view(
			{normPreOutput.size(0) / seqLen, seqLen, normPreOutput.size(1)});

//	Tensor lstmState;
	auto lstm0RnnOutput = gru0->forward(lstmInput);
	auto lstm0Output = std::get<0>(lstm0RnnOutput);

	auto batchNorm0Output = batchNorm0->forward(lstm0Output);
//	lstmState = lstmRnnOutput.state;

	auto lstmOutput = batchNorm0Output;

	Tensor fcOutput = fc->forward(lstmOutput);

	Tensor output = torch::log_softmax(fcOutput, OutputMax);

	return output.view({output.size(0) * output.size(1), output.size(2)});
}


torch::Tensor FcGRUNet::inputPreprocess(torch::Tensor input) {
	return input.div(4);
}

const std::string FcGRUNet::GetName() {
	return "FCGRUNet";
}
