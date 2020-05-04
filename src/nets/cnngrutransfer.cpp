/*
 * cnngrutransfer.cpp
 *
 *  Created on: Apr 7, 2020
 *      Author: zf
 */


/*
 * cnnnet.cpp
 *
 *  Created on: Mar 16, 2020
 *      Author: zf
 */

#include "nets/cnngrutransfer.h"

#include <torch/torch.h>
#include "lmdbtools/LmdbReader.h"
#include "lmdbtools/LmdbDataDefs.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <stdio.h>
#include <experimental/filesystem>

using Tensor = torch::Tensor;
using TensorList = torch::TensorList;
using string = std::string;

CNNGRUTransfer::CNNGRUTransfer(const int inSeqLen, const torch::nn::Module& transferredNet) :
		conv0(torch::nn::Conv2dOptions(1, 32, {5, 3}).padding({2, 1})),
		batchNorm0(32),
		conv1(torch::nn::Conv2dOptions(32, 64, {3, 3}).padding({1, 1})),
		batchNorm1(64),
		conv2(torch::nn::Conv2dOptions(64, 128, {3, 3}).padding({1, 1})),
		batchNorm2(128),
		gru0(torch::nn::GRUOptions(4864, 4864).batch_first(true)),
		batchNormGru0(inSeqLen),
		fc0(torch::nn::Linear(4864, 1024)),
		fcBatchNorm0(1024),
		fc1(torch::nn::Linear(1024, 42)),
		dataFile("./cnnnet.txt"),
		seqLen(inSeqLen)
{
	register_module("conv0", conv0);
	register_module("batchNorm0", batchNorm0);
	register_module("conv1", conv1);
	register_module("batchNorm1", batchNorm1);
	register_module("conv2", conv2);
	register_module("batchNorm2", batchNorm2);
	register_module("gru0", gru0);
	register_module("batchNormGru0", batchNormGru0);
	register_module("fc0", fc0);
	register_module("fcBatchNorm0", fcBatchNorm0);
	register_module("fc1", fc1);

	loadParams(transferredNet);
	initParams();

	std::cout << "After all ------------------------------> " << std::endl;
	auto params = this->named_parameters(true);
	for (auto ite = params.begin(); ite != params.end(); ite ++) {
		std::cout << ite->key() << ": " << ite->value().sizes() << std::endl;
	}
	std::cout << "End of constructor " << std::endl;
}

CNNGRUTransfer::~CNNGRUTransfer() {
	dataFile.close();
}

void CNNGRUTransfer::initParams() {
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

void CNNGRUTransfer::loadParams(const torch::nn::Module& transferredNet) {
	std::cout << "Load params ---------------> " << std::endl;
	auto params = this->named_parameters(true);
	auto transferredParams = transferredNet.named_parameters(true);

	for (auto ite = params.begin(); ite != params.end(); ite ++) {
		auto key = ite->key();
		if (transferredParams.find(key) != nullptr) {
//			ite->value().copy_(transferredParams[key]);
			auto transferData = transferredParams[key].data_ptr<float>();
			auto paramData = ite->value().data_ptr<float>();
			if (ite->value().numel() != transferredParams[key].numel()) {
				std::cout << "Not in same size "
						<< ite->value().sizes() << " --- "
						<< transferredParams[key].sizes()
						<< std::endl;
				continue;
			}

			for (auto i = 0; i < ite->value().numel(); i ++) {
				paramData[i] = transferData[i];
			}
			std::cout << "Copied param " << key << std::endl;
		}
	}

}

torch::Tensor CNNGRUTransfer::inputPreprocess(torch::Tensor input) {
	return input.div(4);
}

void CNNGRUTransfer::setTrain(bool isTrain) {
	conv0->train(isTrain);
	batchNorm0->train(isTrain);
	conv1->train(isTrain);
	batchNorm1->train(isTrain);
	conv2->train(isTrain);
	batchNorm2->train(isTrain);
	gru0->train(isTrain);
	batchNormGru0->train(isTrain);
	fc0->train(isTrain);
	fcBatchNorm0->train(isTrain);
	fc1->train(isTrain);
}

torch::Tensor CNNGRUTransfer::forward(std::vector<torch::Tensor> inputs, const int seqLen, bool isTrain, bool toRecord) {
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
	conv0Output = torch::leaky_relu(conv0Output);
	conv0Output = torch::max_pool2d(conv0Output, {2, 2}, {1, 1}, {1, 1});

	auto batch0Output = batchNorm0->forward(conv0Output);
//	std::cout << "batch0Outpu " << batch0Output.sizes() << std::endl;

	auto conv1Output = conv1->forward(batch0Output);
	conv1Output = torch::leaky_relu(conv1Output);
	conv1Output = torch::max_pool2d(conv1Output, {2, 2}, {2, 2}, {1, 1});

	auto batch1Output = batchNorm1->forward(conv1Output);
//	std::cout << "batch1Outpu " << batch1Output.sizes() << std::endl;

	auto conv2Output = conv2->forward(batch1Output);
	conv2Output = torch::leaky_relu(conv2Output);
	conv2Output = torch::max_pool2d(conv2Output, {2, 2}, {2, 2}, {0, 1});

	auto batch2Output = batchNorm2->forward(conv2Output);
//	std::cout << "batch2Outpu " << batch2Output.sizes() << std::endl;

	auto gruInput = batch2Output;
	gruInput = gruInput.view(
			{gruInput.size(0) / seqLen, seqLen, gruInput.size(1) * gruInput.size(2) * gruInput.size(3)});
	auto gruRnnOutput = gru0->forward(gruInput);
	auto gruOutput = std::get<0>(gruRnnOutput);
//	std::cout << "To run batch gru0 " << std::endl;
	auto batchNormGru0Output = batchNormGru0->forward(gruOutput);
//	std::cout << "End of run batch gru0 " << std::endl;


//	Tensor convOutput = batch2Output;
//	std::cout << "Con0output " << convOutput.sizes() << std::endl;
//	Tensor fcInput = convOutput.view(
//			{convOutput.size(0), convOutput.size(1) * convOutput.size(2) * convOutput.size(3)});
	auto fcInput = batchNormGru0Output;
	fcInput = fcInput.view({fcInput.size(0) * fcInput.size(1), fcInput.size(2)});
	auto fc0Output = fc0->forward(fcInput);
//	std::cout << "End of f0Output " << std::endl;

//	std::cout << "fc0Output sizes: " << fc0Output.sizes() << std::endl;
	auto fc1Input = fcBatchNorm0->forward(fc0Output);
//	std::cout << "End of fcBatchNorm0 " << std::endl;

	auto fc1Output = fc1->forward(fc1Input);
//	std::cout << "End of fc1 " << std::endl;

	auto fcOutput = fc1Output;
	Tensor output = torch::log_softmax(fcOutput, 1);

//	std::cout << "End of forward" << std::endl;
	return output;
}

const std::string CNNGRUTransfer::GetName() {
	return "CNNGRUTransfer";
}




