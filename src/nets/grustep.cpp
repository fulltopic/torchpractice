/*
 * grustep.cpp
 *
 *  Created on: May 14, 2020
 *      Author: zf
 */

#include <torch/torch.h>
#include "lmdbtools/LmdbReader.h"
#include "lmdbtools/LmdbDataDefs.h"
#include "nets/NetDef.h"
#include "nets/grustep.h"

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

thread_local torch::Tensor GRUStepNet::state;
thread_local bool GRUStepNet::resetState = false;

GRUStepNet::GRUStepNet():
		gru0(torch::nn::GRUOptions(360, 1024).batch_first(true)),
		batchNorm0(1),
		fc(1024, FcOutput)
//		batchSize(inBatchSize)
{
//	register_module("lstm0", gru0);
	register_module("gru0", gru0);
	register_module("batchNorm0", batchNorm0);
	register_module("fc", fc);

	initParams();
}

void GRUStepNet::loadParams(const string modelPath) {
	//TODO
}

void GRUStepNet::initParams() {
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

std::vector<Tensor> GRUStepNet::forward(std::vector<Tensor> inputs) {
	return forward(inputs, false);
}

std::vector<Tensor> GRUStepNet::forward(std::vector<Tensor> inputs, bool isTrain) {
//	gru0->train(isTrain);
//	batchNorm0->train(isTrain);
//	fc->train(isTrain);


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

	Tensor output = torch::log_softmax(fcOutput, 2);

//	return output;
	return {output.view({output.size(0) * output.size(1), output.size(2)})};
}

void GRUStepNet::reset() {
	resetState = true;
}


const string GRUStepNet::GetName() {
	return "GRUStep";
}


