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
#include "rltest/rltestutils.h"
#include "rltest/rltestsetting.h"

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

		//TODO: Initialize fc by gymtest utils
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
	fc(2048, 42),
//	fcValue(2048, 1),
	seqLen(inSeqLen)
{
	cout << "l2 constructor" << endl;
	register_module("gru0", gru0);
	register_module("fc", fc);
//	register_module("fcValue", fcValue);

//		initParams();
}

Tensor GRUL2Net::inputPreprocess(Tensor input) {
//	return input.div(4);
	return input;
}

//inputs sorted
//TODO: Caution: inputs are re-ordered
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

//	Tensor logmaxInput = fcOutput;
//	Tensor logmaxOutput = torch::log_softmax(logmaxInput, -1);
//
//	return {logmaxOutput};

	return {fcOutput, fcInput};
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

//vector: inputs, actions, returns
//each vector: {seqLen, others}
// inputTensors[2] = rewards = float[batchSize]
//Tensor GRUL2Net::getLoss(vector<vector<Tensor>> inputTensors){
//	vector<Tensor> inputs = inputTensors[InputIndex];
//	vector<Tensor> actions = inputTensors[ActionIndex];
//	vector<Tensor> rewards = inputTensors[RewardIndex];
//
//	vector<Tensor> returnTensors;
//	for (int i = 0; i < rewards.size(); i ++) {
//		Tensor returnTensor = rltest::Utils::BasicReturnCalc(rewards[i], {}, inputs[i].size(0), 0.99);
//		returnTensors.push_back(returnTensor);
//	}
//
//	std::stable_sort(inputs.begin(), inputs.end(), Utils::compTensorBySeqLen);
//	std::stable_sort(actions.begin(), actions.end(), Utils::compTensorBySeqLen);
//	std::stable_sort(returnTensors.begin(), returnTensors.end(), Utils::compTensorBySeqLen);
//
//	Tensor action = torch::stack(actions, 0);
//	Tensor returns = torch::stack(returnTensors, 0);
//
//
//
//	//TODO: It is not right to just forward {inputs} as inputs of forward requires re
//	vector<Tensor> output = forward(inputs, true);
//	Tensor actionOutput = output[0];
//	Tensor valueOutput = output[1];
//
//	Tensor adv = returns - valueOutput;
//	Tensor valueLoss = 0.5 * adv.pow(2).mean();
//
//	Tensor actionLogProbs = torch::log_softmax(action, -1); //TODO: actionOutput is of fc output
//	Tensor actionProbs = torch::softmax(action, -1);
//	actionProbs = actionProbs.clamp(1.21e-7, 1.0f - 1.21e-7);
//	Tensor entropy = -(actionLogProbs * actionProbs).sum(-1).mean();
//
//	Tensor actPi = actionLogProbs.gather(-1, action);
//	Tensor actionLoss = -(actPi * adv.detach()).mean();
//
//	cout << "valueLoss: " << valueLoss.item<float>() << endl;
//	cout << "actionLoss: " << actionLoss.item<float>() << endl;
//	cout << "entropy: " << entropy.item<float>() << endl;
//	cout << "-----------------------------------------> " << endl;
//
//	Tensor loss = valueLoss + actionLoss - entropy * 1e-4;
//
//	return loss;
//}

Tensor GRUL2Net::createHState() {
	int cellSize =  gru0->options.hidden_size();
	return torch::zeros({gru0->options.num_layers(), 1, cellSize});
}

void GRUL2Net::reset() {
	register_module("gru0", gru0);
	register_module("fc", fc);
}

void GRUL2Net::cloneFrom(const GRUL2Net& origNet) {
//    torch::NoGradGuard no_grad;
//
//    this->parameters_.clear();
//    this->buffers_.clear();
//    this->children_.clear();
//    reset();
//
//
//    const torch::optional<torch::Device>& device = torch::nullopt;
//    TORCH_CHECK(
//        origNet.parameters_.size() == parameters_.size(),
//        "The cloned module does not have the same number of "
//        "parameters as the original module after calling reset(). "
//        "Are you sure you called register_parameter() inside reset() "
//        "and not the constructor?");
//    for (const auto& parameter : origNet.named_parameters(/*recurse=*/true)) {
//      auto& tensor = *parameter;
//      auto data = torch::autograd::Variable(tensor).clone();
//      this->parameters_[parameter.key()].set_data(data);
//    }
//
//    TORCH_CHECK(
//        origNet.buffers_.size() == buffers_.size(),
//        "The cloned module does not have the same number of "
//        "buffers as the original module after calling reset(). "
//        "Are you sure you called register_buffer() inside reset() "
//        "and not the constructor?");
//    for (const auto& buffer : origNet.named_buffers(/*recurse=*/true)) {
//      auto& tensor = *buffer;
//      auto data = torch::autograd::Variable(tensor).clone();
//      this->buffers_[buffer.key()].set_data(data);
//    }
//
//    TORCH_CHECK(
//        origNet.children_.size() == children_.size(),
//        "The cloned module does not have the same number of "
//        "child modules as the original module after calling reset(). "
//        "Are you sure you called register_module() inside reset() "
//        "and not the constructor?");
//    for (const auto& child : origNet.children_) {
//      this->children_[child.key()]->clone_(*child.value(), device);
//    }
}

GRUL2Net::GRUL2Net(const GRUL2Net& net):
	gru0(torch::nn::GRUOptions(360, 2048).num_layers(2).batch_first(true)),
	fc(2048, 42),
//	fcValue(2048, 1),
	seqLen(net.getSeqLen())
{
	cout << "l2 copy constructor" << endl;
	register_module("gru0", gru0);
	register_module("fc", fc);
//	register_module("fcValue", fcValue);

//		initParams();
}

GRUL2Net& GRUL2Net::operator=(const GRUL2Net& other) {
	return *this;
}

GRUL2Net::GRUL2Net(GRUL2Net&& net):
	gru0(torch::nn::GRUOptions(360, 2048).num_layers(2).batch_first(true)),
	fc(2048, 42),
//	fcValue(2048, 1),
	seqLen(net.getSeqLen())
{
	cout << "l2 move constructor " << endl;
	register_module("gru0", gru0);
	register_module("fc", fc);
//	register_module("fcValue", fcValue);

//		initParams();
}

//GRUL2Net& GRUL2Net::operator=(GRUL2Net&& other);
//	return *this;
//}
}


