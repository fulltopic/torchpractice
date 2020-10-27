/*
 * l2rlovernet.cpp
 *
 *  Created on: Sep 14, 2020
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

#include "rltest/rltestutils.h"
#include "rltest/rltestsetting.h"
#include "rltest/gaecalc.h"

#include "rltest/l2rlovernet.h"
#include "utils/logger.h"
#include "utils/storedata.h"

namespace rltest {
using Tensor = torch::Tensor;
using string = std::string;
using std::cout;
using std::endl;
using TensorList = torch::TensorList;
using std::vector;

void GRUL2OverNet::initParams() {
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

void GRUL2OverNet::printGrads() {
	Logger::GetLogger()->info("To print grads ");

	auto params = this->named_parameters(true);
	for (auto ite = params.begin(); ite != params.end(); ite ++) {
		Logger::GetLogger()->info("Get key: {} ", ite->key());
		const Tensor paramTensor = ite->value();
		const Tensor paramGrad = paramTensor.grad();

		auto mean = paramGrad.mean();
		auto var = paramGrad.var();
		auto minTensor = paramGrad.min();
		auto maxTensor = paramGrad.max();

		std::cout << "Mean: ---> " << endl << mean << std::endl;
		cout << "Var: --> " << endl << var << endl;
		cout << "Min: --> " << endl << minTensor << endl;
		cout << "Max: --> " << endl << maxTensor << endl;
	}
}

void GRUL2OverNet::loadL2Model(const std::string modelPath) {
	Logger::GetLogger()->info("To load l2 model ");

	unregister_module("fcValue");
	try {
	torch::serialize::InputArchive inChive;
	inChive.load_from(modelPath);
	this->load(inChive);
	} catch (std::exception& e) {
		Logger::GetLogger()->error("Failed to load as: {}", e.what());
	}

	register_module("fcValue", fcValue);

	Logger::GetLogger()->info("Loaded l2 model");
}

void GRUL2OverNet::loadModel(const std::string modelPath) {
	Logger::GetLogger()->info("To load overall model from {}", modelPath);

	try {
		torch::serialize::InputArchive inChive;
		inChive.load_from(modelPath);
		this->load(inChive);
		Logger::GetLogger()->info("Loaded overall model");
	} catch (std::exception& e) {
		//TODO: Should recover?
		Logger::GetLogger()->error("Failed to load as: {}", e.what());
	}
}


GRUL2OverNet::GRUL2OverNet(int inSeqLen):
	gru0(torch::nn::GRUOptions(360, 2048).num_layers(2).batch_first(true)),
	fc(2048, 42),
	fcValue(2048, 1),
	seqLen(inSeqLen)
{
	Logger::GetLogger()->info("l2 constructor");
	register_module("gru0", gru0);
	register_module("fc", fc);
	register_module("fcValue", fcValue);

//	initParams();
}

GRUL2OverNet::GRUL2OverNet(int inSeqLen, bool isL2Model, const std::string modelPath):
		GRUL2OverNet(inSeqLen)
{
	if (isL2Model) {
		loadL2Model(modelPath);
	} else {
		loadModel(modelPath);
	}
}


Tensor GRUL2OverNet::inputPreprocess(Tensor input) {
//	return input.div(4);
	return input;
}

//inputs sorted
//TODO: Caution: inputs are re-ordered
vector<Tensor> GRUL2OverNet::forward(vector<Tensor> inputs, bool isTrain) {
	vector<int64_t> seqLens(inputs.size(), 0);
	int total = 0;
	for (int i = 0; i < inputs.size(); i ++) {
		seqLens[i] = inputs[i].size(0);
		total += seqLens[i];

		inputs[i] = inputPreprocess(inputs[i]);
		inputs[i] = inputs[i].view({inputs[i].size(0), inputs[i].size(1) * inputs[i].size(2)}); //TODO: -1 is OK?
		inputs[i] = torch::constant_pad_nd(inputs[i], {0, 0, 0, (seqLen - seqLens[i])}); //TODO: remove head or end is better?
		seqLens[i] = std::min(seqLens[i], seqLen);
	}
	Logger::GetLogger()->info("total input {}", total);
//	cout << "total input " << total << endl;

//		auto temp = torch::stack(inputs, 0);
	auto packedInput = torch::nn::utils::rnn::pack_padded_sequence(torch::stack(inputs, 0), torch::tensor(seqLens), true);
	auto gruOutput = gru0->forward_with_packed_input(packedInput);
	auto gruOutputData = std::get<0>(gruOutput);
	Logger::GetLogger()->info("End of gru cell");
//	cout << "End of gru cell " << endl;

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
	Logger::GetLogger()->info("Output total {}", outputTotal);
//	cout << "Output total " << outputTotal << endl;

	Tensor fcInput = torch::cat(unpackDatas, 0);
	auto fcOutput = fc->forward(fcInput);
	Logger::GetLogger()->info("End of fc cell");

	Tensor fcValueOutput = fcValue->forward(fcInput);

//	cout << "End of fc cell " << endl;

//	Tensor logmaxInput = fcOutput;
//	Tensor logmaxOutput = torch::log_softmax(logmaxInput, -1);
//
//	return {logmaxOutput};

	return {fcOutput, fcValueOutput};
}


vector<Tensor> GRUL2OverNet::forward (vector<Tensor> inputs) {
	Tensor input = inputs[0];
	Tensor hState = inputs[1];
	const int step = inputs[2].item<long>();
	Logger::GetLogger()->info("----------------------> step: {}", step);
//	cout << "-------------------> step: " << step << endl;

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
//Tensor GRUL2OverNet::getLoss(vector<vector<Tensor>> inputTensors){
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

Tensor GRUL2OverNet::createHState() {
	int cellSize =  gru0->options.hidden_size();
	return torch::zeros({gru0->options.num_layers(), 1, cellSize});
}

void GRUL2OverNet::reset() {
	register_module("gru0", gru0);
	register_module("fc", fc);
	register_module("fcValue", fcValue);
}

void GRUL2OverNet::cloneFrom(const GRUL2OverNet& origNet) {
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

GRUL2OverNet::GRUL2OverNet(const GRUL2OverNet& net):
	gru0(torch::nn::GRUOptions(360, 2048).num_layers(2).batch_first(true)),
	fc(2048, 42),
	fcValue(2048, 1),
	seqLen(net.getSeqLen())
{
	Logger::GetLogger()->info("l2 copy constructor");
//	cout << "l2 copy constructor" << endl;
	register_module("gru0", gru0);
	register_module("fc", fc);
	register_module("fcValue", fcValue);

	//TODO
//	initParams();
}

GRUL2OverNet& GRUL2OverNet::operator=(const GRUL2OverNet& other) {
	return *this;
}

GRUL2OverNet::GRUL2OverNet(GRUL2OverNet&& net):
	gru0(torch::nn::GRUOptions(360, 2048).num_layers(2).batch_first(true)),
	fc(2048, 42),
	fcValue(2048, 1),
	seqLen(net.getSeqLen())
{
//	cout << "l2 move constructor " << endl;
	Logger::GetLogger()->info("l2 move constructor");
	register_module("gru0", gru0);
	register_module("fc", fc);
	register_module("fcValue", fcValue);

//	initParams();
}

//GRUL2OverNet& GRUL2OverNet::operator=(GRUL2OverNet&& other);
//	return *this;
//}

vector<Tensor> GRUL2OverNet::getLoss(vector<vector<Tensor>> inputTensors){
	GAECal calc(0.99, 0.95);

	this->train();

	vector<Tensor> inputs = inputTensors[InputIndex];
	vector<Tensor> actions = inputTensors[ActionIndex];
	vector<Tensor> rewards = inputTensors[RewardIndex];
	vector<Tensor> labels = inputTensors[LabelIndex];
	const int batchSize = inputs.size();

	cout << "Input sizes: " << inputs[0].sizes() << endl;
	cout << "action sizes: " << actions[0].sizes() << endl;
	cout << "reward sizes: " << rewards[0].sizes() << endl;
	cout << "label sizes: " << labels[0].sizes() << endl;

	cout << "label games: " << labels.size() << endl;
	cout << "reward games: " << rewards.size() << endl;
	cout << "reward sizes: " << rewards[0].sizes() << endl;
//	vector<Tensor> returnTensors;
	vector<Tensor> advs;

	struct SeqReward {
		int seqLen;
		Tensor reward;
	};
	auto sortSeqReward = [](const SeqReward& t0, const SeqReward& t1) {
		return t0.seqLen > t1.seqLen;
	};

	vector<SeqReward> seqRewards;
	vector<int> seqLens;
	for (int i = 0; i < rewards.size(); i ++) {
		actions[i] = torch::constant_pad_nd(actions[i], {0, 0, 0, std::min(seqLen, actions[i].size(0)) - actions[i].size(0)});
		labels[i] = torch::constant_pad_nd(labels[i], {0, std::min(seqLen, labels[i].size(0)) - labels[i].size(0)});
		labels[i] = labels[i].view({labels[i].size(0), 1});
		cout << "action len: " << actions[i].sizes() << endl;
		cout << "labels len: " << labels[i].sizes() << endl;
		seqLens.push_back(std::min((int)inputs[i].size(0), (int)seqLen));
		seqRewards.push_back(SeqReward {std::min((int)inputs[i].size(0), (int)seqLen), rewards[i]});
//		cout << "reward " << i << " sizes: " << rewards[i].sizes() << endl;
	}

	std::stable_sort(inputs.begin(), inputs.end(), Utils::CompTensorBySeqLen);
	std::stable_sort(actions.begin(), actions.end(), Utils::CompTensorBySeqLen);
	std::stable_sort(labels.begin(), labels.end(), Utils::CompTensorBySeqLen);
	std::stable_sort(seqRewards.begin(), seqRewards.end(), sortSeqReward);
//	std::stable_sort(returnTensors.begin(), returnTensors.end(), Utils::CompTensorBySeqLen);

	Tensor action = torch::cat(actions, 0);
	Tensor label = torch::cat(labels, 0);
//	Tensor returns = torch::cat(returnTensors, 0);


	vector<Tensor> output = forward(inputs, true);
	Tensor actionOutput = output[0];
	Tensor valueOutput = output[1];

//	cout << "return sizes: " << returns.sizes() << endl;
	cout << "value sizes: " << valueOutput.sizes() << endl;
	cout << "actionOutput sizes: " << actionOutput.sizes() << endl;
	cout << "valueOutput sizes: " << valueOutput.sizes() << endl;

	int endIndex = 0;
	vector<Tensor> returnValues;
	vector<Tensor> advValues;
	for (int i = 0; i < batchSize; i ++) {
		Tensor valueInput =  valueOutput.slice(0, endIndex, endIndex + seqLens[i]);
		endIndex += seqLens[i];

		Tensor returnTensor = calc.calc(valueInput, seqRewards[i].reward).detach(); //TODO: Why detach?
		returnValues.push_back(returnTensor);
		Tensor advTensor = calc.calcAdv(valueInput, returnTensor);
		advValues.push_back(advTensor);
	}
	//	cout << "return sizes: " << returnValues[0].sizes() << endl;
	Tensor adv = torch::cat(advValues, 0);
	cout << "adv " << endl << adv << endl;
	adv = adv.view({adv.size(0), 1});


	Tensor valueLoss = 0.5 * adv.pow(2).mean();

	Tensor actionLogProbs = torch::log_softmax(actionOutput, -1); //actionOutput is of fc output
	Tensor actionProbs = torch::softmax(actionOutput, -1);
	actionProbs = actionProbs.clamp(1.21e-7, 1.0f - 1.21e-7);
	Tensor entropy = -(actionLogProbs * actionProbs).sum(-1).mean();

	Tensor actPi = actionLogProbs.gather(-1, label);
	Tensor actionLoss = -(actPi * adv.detach()).mean();
	cout << "actionLog sizes " << actionLogProbs.sizes() << " ========== " << "actPi sizes " << actPi.sizes() << std::endl;

	Logger::GetLogger()->info("valueLoss: {}", valueLoss.item<float>());
	Logger::GetLogger()->info("actionLoss: {}", actionLoss.item<float>());
	Logger::GetLogger()->info("entropy: {}", entropy.item<float>());
	Logger::GetLogger()->info("-----------------------------------------> ");

	Tensor loss = valueLoss + actionLoss - entropy * 1e-4;

//	return loss;
	return {loss, valueLoss, actionLoss, entropy};
}

//Tensor GRUL2OverNet::getLoss(vector<vector<Tensor>> inputTensors){
//	this->train();
//
//	vector<Tensor> inputs = inputTensors[InputIndex];
//	vector<Tensor> actions = inputTensors[ActionIndex];
//	vector<Tensor> rewards = inputTensors[RewardIndex];
//	vector<Tensor> labels = inputTensors[LabelIndex];
//
//	cout << "Input sizes: " << inputs[0].sizes() << endl;
//	cout << "action sizes: " << actions[0].sizes() << endl;
//	cout << "reward sizes: " << rewards[0].sizes() << endl;
//	cout << "label sizes: " << labels[0].sizes() << endl;
//
//	cout << "label games: " << labels.size() << endl;
//	cout << "reward games: " << rewards.size() << endl;
//	vector<Tensor> returnTensors;
//	for (int i = 0; i < rewards.size(); i ++) {
//		Tensor returnTensor = rltest::Utils::BasicReturnCalc(
//				rewards[i], labels[i], actions[i], actions[i].size(0), RlSetting::ReturnGamma, 0.0f);
//
//		returnTensor = torch::constant_pad_nd(returnTensor, {0, 0, 0, std::min(seqLen, actions[i].size(0)) - actions[i].size(0)});
//		actions[i] = torch::constant_pad_nd(actions[i], {0, 0, 0, std::min(seqLen, actions[i].size(0)) - actions[i].size(0)});
//		labels[i] = torch::constant_pad_nd(labels[i], {0, std::min(seqLen, actions[i].size(0)) - actions[i].size(0)});
//
////		cout << "action len: " << actions[i].sizes() << endl;
////		cout << "return len: " << returnTensor.sizes() << endl;
//		returnTensors.push_back(returnTensor);
//	}
//	cout << "return sizes: " << returnTensors[0].sizes() << endl;
//
//	std::stable_sort(inputs.begin(), inputs.end(), Utils::CompTensorBySeqLen);
//	std::stable_sort(actions.begin(), actions.end(), Utils::CompTensorBySeqLen);
//	std::stable_sort(labels.begin(), labels.end(), Utils::CompTensorBySeqLen);
//	std::stable_sort(returnTensors.begin(), returnTensors.end(), Utils::CompTensorBySeqLen);
//
//	Tensor action = torch::cat(actions, 0);
//	Tensor label = torch::cat(labels, 0);
//	label = label.view({label.size(0), 1});
//	Tensor returns = torch::cat(returnTensors, 0);
//
//
//
//	//TODO: It is not right to just forward {inputs} as inputs of forward
//	vector<Tensor> output = forward(inputs, true);
//	Tensor actionOutput = output[0];
//	Tensor valueOutput = output[1];
//
//	cout << "return sizes: " << returns.sizes() << endl;
//	cout << "value sizes: " << valueOutput.sizes() << endl;
//	cout << "actionOutput sizes: " << actionOutput.sizes() << endl;
//	cout << "valueOutput sizes: " << valueOutput.sizes() << endl;
//	Tensor adv = returns - valueOutput;
//	Tensor valueLoss = 0.5 * adv.pow(2).mean();
//
//	Tensor actionLogProbs = torch::log_softmax(actionOutput, -1); //TODO: actionOutput is of fc output
//	Tensor actionProbs = torch::softmax(actionOutput, -1);
//	actionProbs = actionProbs.clamp(1.21e-7, 1.0f - 1.21e-7);
//	Tensor entropy = -(actionLogProbs * actionProbs).sum(-1).mean();
//
//	Tensor actPi = actionLogProbs.gather(-1, label);
//	Tensor actionLoss = -(actPi * adv.detach()).mean();
//	cout << "actionLog sizes " << actionLogProbs.sizes() << " ========== " << "actPi sizes " << actPi.sizes() << std::endl;
//
//	Logger::GetLogger()->info("valueLoss: {}", valueLoss.item<float>());
//	Logger::GetLogger()->info("actionLoss: {}", actionLoss.item<float>());
//	Logger::GetLogger()->info("entropy: {}", entropy.item<float>());
//	Logger::GetLogger()->info("-----------------------------------------> ");
//
//	Tensor loss = valueLoss + actionLoss - entropy * 1e-4;
//
//	return loss;
//}

std::string GRUL2OverNet::GetName() {
	return "GRUL2OverNet";
}
}


