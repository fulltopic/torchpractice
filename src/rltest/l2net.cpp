/*
 * l2net.cpp
 *
 *  Created on: Sep 4, 2020
 *      Author: zf
 */


#include <string.h>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <stdexcept>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <torch/torch.h>

//#include <cpprl/cpprl.h>
//#include <matplotlibcpp.h>
//#include "pytools/syncplotserver.h"

#include "lmdbtools/LmdbReaderWrapper.h"
#include "lmdbtools/LmdbReader.h"
#include "lmdbtools/LmdbDataDefs.h"
#include "lmdbtools/Lmdb2RowDataDefs.h"

#include "rltest/l2net.h"

namespace rltest {
using Tensor = torch::Tensor;
using string = std::string;
using std::cout;
using std::endl;
using TensorList = torch::TensorList;
using std::vector;

void GRUL2Net::initParams() {
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

void GRUL2Net::loadModel(const std::string modelPath) {
	torch::serialize::InputArchive inChive;
	inChive.load_from(modelPath);
	this->load(inChive);
}


GRUL2Net::GRUL2Net(int inSeqLen):
	gru0(torch::nn::GRUOptions(360, 2048).num_layers(2).batch_first(true)),
	fc(2048, 1),
	seqLen(inSeqLen)
{
	register_module("gru0", gru0);
	register_module("fc", fc);

//		initParams();
}

Tensor GRUL2Net::inputPreprocess(Tensor input) {
//	return input.div(4);
	return input;
}

vector<Tensor> GRUL2Net::forward(vector<Tensor> inputs, bool isTrain) {
	vector<int> seqLens(inputs.size(), 0);
	int total = 0;
	for (int i = 0; i < inputs.size(); i ++) {
		seqLens[i] = inputs[i].size(0);
		total += seqLens[i];

		inputs[i] = inputPreprocess(inputs[i]);
		inputs[i] = inputs[i].view({inputs[i].size(0), inputs[i].size(1) * inputs[i].size(2)}); //TODO: -1 is OK?
		inputs[i] = torch::constant_pad_nd(inputs[i], {0, 0, 0, (seqLen - seqLens[i])});
		seqLens[i] = std::min(seqLens[i], seqLen);
	}
	cout << "total input " << total << endl;

//		auto temp = torch::stack(inputs, 0);
	auto packedInput = torch::nn::utils::rnn::pack_padded_sequence(torch::stack(inputs, 0), torch::tensor(seqLens), true);
	auto gruOutput = gru0->forward_with_packed_input(packedInput);
	auto gruOutputData = std::get<0>(gruOutput);
	cout << "End of gru cell " << endl;

	auto unpacked = torch::nn::utils::rnn::pad_packed_sequence(gruOutputData, true);
	Tensor unpackedTensor = std::get<0>(unpacked);
	cout << "unpackedTensor: " << unpackedTensor.sizes() << endl;
	vector<Tensor> unpackDatas(seqLens.size());
	int outputTotal = 0;
	for (int i = 0; i < seqLens.size(); i ++) {
		Tensor unpackData = unpackedTensor[i].narrow(0, 0, seqLens[i]);
		unpackDatas[i] = unpackData;
		outputTotal += (int)unpackDatas[i].size(0);
	}
	cout << "Output total " << outputTotal << endl;

	Tensor fcInput = torch::cat(unpackDatas, 0);
	auto fcOutput = fc->forward(fcInput);
	cout << "End of fc cell " << endl;

	Tensor logmaxInput = fcOutput;
	Tensor logmaxOutput = torch::log_softmax(logmaxInput, -1);

	return {logmaxOutput};
}


vector<Tensor> GRUL2Net::forward (vector<Tensor> inputs) {
	Tensor input = inputs[0];
	Tensor hState = inputs[1];
	const int step = inputs[2].item<long>();
	cout << "-------------------> step: " << step << endl;

	if (input.dim() < 3) {
		input = input.view({1, input.size(0), input.size(1)});
	}
	input = inputPreprocess(input);
//		if (hState.dim() < 3) {
//			hState = hState.view({1, hState.size(0), hState.size(1)});
//		}
	auto gruOutputData = gru0->forward(input, hState);
	auto gruOutput = std::get<0>(gruOutputData);
	auto newState = std::get<1>(gruOutputData);


	Tensor fcOutput = fc->forward(gruOutput);
//	cout << "fcOutput: " << fcOutput.sizes() << endl;
	Tensor logmaxOutput = torch::log_softmax(fcOutput, -1);
	logmaxOutput = logmaxOutput.squeeze(1);

	return {logmaxOutput, newState};
}

Tensor GRUL2Net::createHState() {
	int cellSize =  gru0->options.hidden_size();
	return torch::zeros({gru0->options.num_layers(), 1, cellSize});
}

void GRUL2Net::reset() {

}


}

