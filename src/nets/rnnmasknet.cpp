/*
 * rnnmasknet.cpp
 *
 *  Created on: May 23, 2020
 *      Author: zf
 */



#include "nets/rnnmasknet.h"
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

using torch::Tensor;
using std::string;
using std::vector;
using std::cout;
using std::endl;

static bool rnnmaskcomp(const Tensor& t0, const Tensor& t1) {
	return t0.size(0) > t1.size(0);
}

GRUMaskNet::GRUMaskNet(int inSeqLen):
		gru0(torch::nn::GRUOptions(360, 1024).batch_first(true)),
		batchNorm0(inSeqLen),
		fc(1024, FcOutput),
		seqLen(inSeqLen)
{
	register_module("gru0", gru0);
	register_module("batchNorm0", batchNorm0);
	register_module("fc", fc);

	initParams();
}

void GRUMaskNet::initParams() {
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
				auto dataTensor = torch::randn(chunks[i].sizes());
				auto meanTensor = torch::mean(dataTensor);
				std::cout << "Mean " << meanTensor.item<float>() << std::endl;
				auto varTensor = torch::var(dataTensor);
				std::cout << "Var " << varTensor.item<float>() << std::endl;
				dataTensor = dataTensor.div(sqrt((double)chunkSize));

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

//Tensor GRUMaskNet::forward(vector<Tensor> inputs, bool isBatch) {
////	std::sort(inputs.begin(), inputs.end(), rnnmaskcomp);
//	vector<int> seqLens(inputs.size(), 0);
////	for (int i = 0; i < inputs.size(); i ++) {
////		seqLens[i] = inputs[i].size(0);
////	}
////	cout << "Prepare inputs " << endl;
//	for (int i = 0; i < inputs.size(); i ++) {
//		seqLens[i] = inputs[i].size(0);
////		cout << "seqLen: " << seqLens[i] << endl;
//		inputs[i] = inputPreprocess(inputs[i]);
//		inputs[i] = inputs[i].view({inputs[i].size(0), inputs[i].size(1) * inputs[i].size(2)});
////		cout << "changed view " << inputs[i].sizes() << endl;
//		inputs[i] = torch::constant_pad_nd(inputs[i], {0, 0, 0, (seqLen - seqLens[i])});
//		seqLens[i] = std::min(seqLens[i], seqLen);
////		cout << "padded input " << seqLens[i] << "--> " <<  inputs[i].sizes() << endl;
//	}
////	cout << "Padded input " << endl;
//	auto temp = torch::stack(inputs, 0);
////	cout << "Get tmp " << temp.sizes() << endl;
//	auto packedInput = torch::nn::utils::rnn::pack_padded_sequence(torch::stack(inputs, 0), torch::tensor(seqLens), true);
////	cout << "End of packed input " << packedInput.data().sizes() << endl;
//
//	auto gruOutput = gru0->forward_with_packed_input(packedInput);
//	auto gruOutputData = std::get<0>(gruOutput);
//	cout << "gruOutputData " << gruOutputData.data().sizes() << endl;
//
//	auto paddedBatchNormInput = torch::nn::utils::rnn::pad_packed_sequence(gruOutputData, true, 0.0f, seqLen);
//	auto batchNormInput = std::get<0>(paddedBatchNormInput);
////	cout << "batchNormInput " << batchNormInput.sizes() << endl;
//	auto batchNormOutput = batchNorm0->forward(batchNormInput);
//
////	Tensor batchNormOutput = gruOutputData.data();
//	auto fcInputPacked = torch::nn::utils::rnn::pack_padded_sequence(batchNormOutput, torch::tensor(seqLens), true);
//
//	auto fcOutput = fc->forward(fcInputPacked.data());
////	auto fcOutput = fc->forward(batchNormOutput);
////	auto fcOutput = fc->forward(batchNormInput);
//
////	auto packedFcOutput = torch::nn::utils::rnn::pack_padded_sequence(fcOutput, torch::tensor(seqLens), true);
//
////	Tensor logmaxInput = packedFcOutput.data();
//	Tensor logmaxInput = fcOutput;
////	cout << "logmaxInput input: " << logmaxInput.sizes() << endl;
//	Tensor logmaxOutput = torch::log_softmax(logmaxInput, 1);
//
//	return logmaxOutput;
//}

//Tensor GRUMaskNet::forward(vector<Tensor> inputs, bool isBatch) {
////	std::sort(inputs.begin(), inputs.end(), rnnmaskcomp);
//	vector<int> seqLens(inputs.size(), 0);
////	for (int i = 0; i < inputs.size(); i ++) {
////		seqLens[i] = inputs[i].size(0);
////	}
////	cout << "Prepare inputs " << endl;
//	for (int i = 0; i < inputs.size(); i ++) {
//		seqLens[i] = inputs[i].size(0);
////		cout << "seqLen: " << seqLens[i] << endl;
//		inputs[i] = inputPreprocess(inputs[i]);
//		inputs[i] = inputs[i].view({inputs[i].size(0), inputs[i].size(1) * inputs[i].size(2)});
////		cout << "changed view " << inputs[i].sizes() << endl;
//		inputs[i] = torch::constant_pad_nd(inputs[i], {0, 0, 0, (seqLen - seqLens[i])});
//		seqLens[i] = std::min(seqLens[i], seqLen);
////		cout << "padded input " << seqLens[i] << "--> " <<  inputs[i].sizes() << endl;
//	}
////	cout << "Padded input " << endl;
//	auto temp = torch::stack(inputs, 0);
////	cout << "Get tmp " << temp.sizes() << endl;
//	auto packedInput = torch::nn::utils::rnn::pack_padded_sequence(torch::stack(inputs, 0), torch::tensor(seqLens), true);
////	cout << "End of packed input " << packedInput.data().sizes() << endl;
//
//	auto gruOutput = gru0->forward_with_packed_input(packedInput);
//	auto gruOutputData = std::get<0>(gruOutput);
//	cout << "gruOutputData " << gruOutputData.data().sizes() << endl;
//
//	auto unpacked = torch::nn::utils::rnn::pad_packed_sequence(gruOutputData, true);
//	Tensor unpackedTensor = std::get<0>(unpacked);
//	vector<Tensor> unpackDatas(seqLens.size());
//	for (int i = 0; i < seqLens.size(); i ++) {
//		Tensor unpackData = unpackedTensor[i].narrow(0, 0, seqLens[i]);
////		unpackDatas.push_back(unpackData);
//		unpackDatas[i] = unpackData;
//	}
//
//	Tensor fcInput = torch::cat(unpackDatas, 0);
//
//	auto fcOutput = fc->forward(fcInput);
//
//	Tensor logmaxInput = fcOutput;
////	cout << "logmaxInput input: " << logmaxInput.sizes() << endl;
//	Tensor logmaxOutput = torch::log_softmax(logmaxInput, 1);
//
//	return logmaxOutput;
//}

Tensor GRUMaskNet::forward(vector<Tensor> inputs, bool isBatch) {
	vector<int> seqLens(inputs.size(), 0);
	for (int i = 0; i < inputs.size(); i ++) {
		seqLens[i] = inputs[i].size(0);
		inputs[i] = inputPreprocess(inputs[i]);
		inputs[i] = inputs[i].view({inputs[i].size(0), inputs[i].size(1) * inputs[i].size(2)});
		inputs[i] = torch::constant_pad_nd(inputs[i], {0, 0, 0, (seqLen - seqLens[i])});
		seqLens[i] = std::min(seqLens[i], seqLen);
	}
	auto temp = torch::stack(inputs, 0);
	auto packedInput = torch::nn::utils::rnn::pack_padded_sequence(torch::stack(inputs, 0), torch::tensor(seqLens), true);
	cout << "End of packed input " << temp.sizes() << endl;

	auto gruOutput = gru0->forward_with_packed_input(packedInput);
	auto gruOutputData = std::get<0>(gruOutput);
	cout << "gruOutputData " << gruOutputData.data().sizes() << endl;

	auto unpacked = torch::nn::utils::rnn::pad_packed_sequence(gruOutputData, true, 0.0f, seqLen);
	Tensor unpackedTensor = std::get<0>(unpacked);

	auto batchInput = unpackedTensor;
	auto batchOutput = batchNorm0->forward(batchInput);
//	cout << "batch output: " << batchOutput.sizes() << endl;


	vector<Tensor> unpackDatas(seqLens.size());
	for (int i = 0; i < seqLens.size(); i ++) {
		Tensor unpackData = batchOutput[i].narrow(0, 0, seqLens[i]);
//		unpackDatas.push_back(unpackData);
		unpackDatas[i] = unpackData;
	}

	Tensor fcInput = torch::cat(unpackDatas, 0);

	auto fcOutput = fc->forward(fcInput);

	Tensor logmaxInput = fcOutput;
//	cout << "logmaxInput input: " << logmaxInput.sizes() << endl;
	Tensor logmaxOutput = torch::log_softmax(logmaxInput, 1);

	return logmaxOutput;
}

Tensor GRUMaskNet::forward(vector<Tensor> inputs, const int seqLen, bool isTrain, bool toRecord) {
	return forward(inputs, true);
}

Tensor GRUMaskNet::inputPreprocess(Tensor input) {
	return input.div(4);
}

const string GRUMaskNet::GetName() {
	return "GRUMaskBatchNet";
}



