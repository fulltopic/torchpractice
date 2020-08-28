/*
 * grutransstep.cpp
 *
 *  Created on: May 20, 2020
 *      Author: zf
 */




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
#include "nets/supervisednet/grutransstep.h"

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

thread_local torch::Tensor GRUTransStepNet::state;
thread_local bool GRUTransStepNet::resetState = false;
thread_local int GRUTransStepNet::step = 0;

GRUTransStepNet::GRUTransStepNet():
		gru0(torch::nn::GRUOptions(360, 1024).batch_first(true)),
		batchNorm0(10),
		fc(1024, FcOutput)
//		batchSize(inBatchSize)
{
//	register_module("lstm0", gru0);
	register_module("gru0", gru0);
	register_module("batchNorm0", batchNorm0);
	register_module("fc", fc);

	initParams();

	for (int i = 0; i < batchNorm0->options.num_features(); i ++) {
		torch::nn::BatchNorm1d stepBatch(1);
		stepBatchNorms.push_back(stepBatch);
	}
}

void GRUTransStepNet::loadParams(const string modelPath) {
	std::cout << "Load model " << modelPath << std::endl;

	torch::serialize::InputArchive inChive;
	inChive.load_from(modelPath);
	this->load(inChive);

//	CNNGRUTransfer net(seqLen, loadNet);


	torch::OrderedDict<std::string, torch::Tensor>  params = this->named_parameters(true);
	for (auto ite = params.begin(); ite != params.end(); ite ++) {
		std::cout << ite->key() << ": " << ite->value().sizes() << std::endl;
	}

	auto batchParams = batchNorm0->parameters(true);
	auto namedParams = batchNorm0->named_buffers(true);
	for (auto nParam: namedParams) {
		std::cout << "named params: " << nParam.key() << ": " << nParam.value().sizes() << std::endl;
	}
	for (auto param: batchParams) {
		std::cout << "Param " << param.sizes() << std::endl;
	}
	std::cout << "Batchnorm weights: " << batchNorm0->weight.sizes() << std::endl;
	std::cout << "Batchnorm bias: " << batchNorm0->bias.sizes() << std::endl;

	std::cout << "bm means " << batchNorm0->running_mean << std::endl;
	std::cout << "bm vars " << batchNorm0->running_var << std::endl;
	std::cout << "bm weights " << batchNorm0->weight << std::endl;
	std::cout << "bm bias " << batchNorm0->bias << std::endl;

	auto wPtr = batchNorm0->weight.accessor<float, 1>();
	auto bPtr = batchNorm0->bias.accessor<float, 1>();

	for (int i = 0; i < stepBatchNorms.size(); i ++) {
		auto bWPtr = stepBatchNorms[i]->weight.accessor<float, 1>();
		auto bBPtr = stepBatchNorms[i]->bias.accessor<float, 1>();
		bWPtr[0] = wPtr[i];
		bBPtr[0] = bPtr[i];

		stepBatchNorms[i]->eval();
	}
}

void GRUTransStepNet::initParams() {
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

//TODO: What default for isBatch?
Tensor GRUTransStepNet::forward(std::vector<Tensor> inputs) {
	return forward(inputs, false);
}

Tensor GRUTransStepNet::forwardBatch(std::vector<Tensor>& inputs) {
	//TODO
	return torch::rand({1, 42});
}

Tensor GRUTransStepNet::forwardStep(std::vector<Tensor>& inputs) {
	std::cout << "BatchNorm sizes " << std::endl;


	Tensor input = inputs[0];
	if (resetState) {
		state = torch::zeros({1, input.size(0), 1024});
		resetState = false;
	}

	auto gruOutput = gru0->forward(input, state);
	auto batchInput = std::get<0>(gruOutput);
	state = std::get<1>(gruOutput);
	Tensor fcInput;
	if (step < batchNorm0->weight.size(0)) {
//	if (step < 0) {
//		torch::nn::BatchNorm1d batchNorm(1);
//
//		auto wPtr = batchNorm0->weight.accessor<float, 1>();
//		auto bPtr = batchNorm0->bias.accessor<float, 1>();
//		auto bWPtr = batchNorm->weight.accessor<float, 1>();
//		auto bBPtr = batchNorm->bias.accessor<float, 1>();
//		bWPtr[0] = wPtr[step];
//		bBPtr[0] = bPtr[step];


//		auto params = batchNorm0->parameters(true);
//		auto meanPtr = params[0].accessor<float, 1>();
//		auto varPtr = params[1].accessor<float, 1>();
//		auto bMeanPtr = batchNorm->running_mean.accessor<float, 1>();
//		auto bVarPtr = batchNorm->running_var.accessor<float, 1>();
//		bMeanPtr[0] = meanPtr[step];
//		bVarPtr[0] = varPtr[step];
//
		fcInput = stepBatchNorms[step]->forward(batchInput);
		step ++;

//		std::cout << "batch norm constructed " << std::endl;
//		std::cout << batchNorm->running_mean.sizes() << std::endl;
//		std::cout << batchNorm->running_var.sizes() << std::endl;

//		fcInput = batchNorm->forward(batchInput);
	} else {
		//nothing
		fcInput = batchInput;
	}
	auto fcOutput = fc->forward(fcInput);
	Tensor output = torch::log_softmax(fcOutput, 2);

//	std::cout << "-------------------------> Output " << std::endl;
//	std::cout << output << std::endl;

	return output.view({output.size(0) * output.size(1), output.size(2)});

//	auto rc = torch::rand({1, 42});
//	return rc;
}

Tensor GRUTransStepNet::forward(std::vector<Tensor> inputs, bool isBatch) {
	std::cout << "-------------------------> isBatch: " << isBatch << std::endl;
	//input sizes = {batch, 1, 360}
//	//state sizes = {1, batch, 1024}
//	Tensor input = inputs[0];
//
//	if (resetState) {
//		state = torch::zeros({1, input.size(0), 1024});
//		resetState = false;
//	}
//
//	auto gruOutput = gru0->forward(input, state);
//	auto batchInput = std::get<0>(gruOutput);
//	state = std::get<1>(gruOutput);
//
//	auto batchOutput = batchNorm0->forward(batchInput);
//	auto fcOutput = fc->forward(batchOutput);
//
//	Tensor output = torch::log_softmax(fcOutput, 2);
//
//	return output.view({output.size(0) * output.size(1), output.size(2)});
	if (isBatch) {
		return forwardBatch(inputs);
	} else {
		return forwardStep(inputs);
	}

}

void GRUTransStepNet::reset() {
	resetState = true;
	step = 0;
}


const string GRUTransStepNet::GetName() {
	return "GRUTransStep";
}


