/*
 * l2rlnet.cpp
 *
 *  Created on: Sep 9, 2020
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

#include "rltest/l2rlcombnet.h"
//#include <cpprl/cpprl.h>
//#include <matplotlibcpp.h>
//#include "pytools/syncplotserver.h"

#include "lmdbtools/LmdbReaderWrapper.h"
#include "lmdbtools/LmdbReader.h"
#include "lmdbtools/LmdbDataDefs.h"
#include "lmdbtools/Lmdb2RowDataDefs.h"

#include "rltest/rltestutils.h"
#include "rltest/rltestsetting.h"

#include "utils/storedata.h"

namespace rltest {
using Tensor = torch::Tensor;
using string = std::string;
using std::cout;
using std::endl;
using TensorList = torch::TensorList;
using std::vector;

void GRUL2RLCombNet::initParams() {
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


void GRUL2RLCombNet::loadModel(const std::string modelPath) {
	l2Net.loadModel(modelPath);
}

GRUL2RLCombNet::GRUL2RLCombNet(int inSeqLen):
	l2Net(inSeqLen),
	fcValue(2048, 1),
	seqLen(inSeqLen)
{
	cout << "rl constructor" << endl;
	register_module("fcValue", fcValue);
	initParams();
}


Tensor GRUL2RLCombNet::inputPreprocess(Tensor input) {
//	return input.div(4);
	return input;
}

//inputs sorted
//TODO: Caution: inputs are re-ordered
vector<Tensor> GRUL2RLCombNet::forward(vector<Tensor> inputs, bool isTrain) {
	vector<Tensor> l2NetOutput = l2Net.forward(inputs, isTrain);
	Tensor fcOutput = l2NetOutput[0];
	Tensor fcInput = l2NetOutput[1];

	auto fcValueOutput = fcValue->forward(fcInput);
	cout << "End of fc cell " << endl;

	return {fcOutput, fcValueOutput};
}


vector<Tensor> GRUL2RLCombNet::forward (vector<Tensor> inputs) {
	return l2Net.forward(inputs);
}

//vector: inputs, actions, returns
//each vector: {seqLen, others}
// inputTensors[2] = rewards = float[batchSize]
Tensor GRUL2RLCombNet::getLoss(vector<vector<Tensor>> inputTensors){
	this->train();

	vector<Tensor> inputs = inputTensors[InputIndex];
	vector<Tensor> actions = inputTensors[ActionIndex];
	vector<Tensor> rewards = inputTensors[RewardIndex];

	vector<Tensor> returnTensors;
	for (int i = 0; i < rewards.size(); i ++) {
		Tensor returnTensor = rltest::Utils::BasicReturnCalc(rewards[i], actions[i], actions[i], inputs[i].size(0), RlSetting::ReturnGamma, 0);
		returnTensors.push_back(returnTensor);
	}

	std::stable_sort(inputs.begin(), inputs.end(), Utils::CompTensorBySeqLen);
	std::stable_sort(actions.begin(), actions.end(), Utils::CompTensorBySeqLen);
	std::stable_sort(returnTensors.begin(), returnTensors.end(), Utils::CompTensorBySeqLen);

	Tensor action = torch::stack(actions, 0);
	Tensor returns = torch::stack(returnTensors, 0);



	//TODO: It is not right to just forward {inputs} as inputs of forward requires re
	vector<Tensor> output = forward(inputs, true);
	Tensor actionOutput = output[0];
	Tensor valueOutput = output[1];

	Tensor adv = returns - valueOutput;
	Tensor valueLoss = 0.5 * adv.pow(2).mean();

	Tensor actionLogProbs = torch::log_softmax(action, -1); //TODO: actionOutput is of fc output
	Tensor actionProbs = torch::softmax(action, -1);
	actionProbs = actionProbs.clamp(1.21e-7, 1.0f - 1.21e-7);
	Tensor entropy = -(actionLogProbs * actionProbs).sum(-1).mean();

	Tensor actPi = actionLogProbs.gather(-1, action);
	Tensor actionLoss = -(actPi * adv.detach()).mean();

	cout << "valueLoss: " << valueLoss.item<float>() << endl;
	cout << "actionLoss: " << actionLoss.item<float>() << endl;
	cout << "entropy: " << entropy.item<float>() << endl;
	cout << "-----------------------------------------> " << endl;

	Tensor loss = valueLoss + actionLoss - entropy * 1e-4;

	return loss;
}

Tensor GRUL2RLCombNet::createHState() {
	return std::move(l2Net.createHState());
}

void GRUL2RLCombNet::reset() {
	l2Net.reset();
	register_module("fcValue", fcValue);
}

GRUL2RLCombNet::GRUL2RLCombNet(const GRUL2RLCombNet& other)
:
			l2Net(other.l2Net),
			fcValue(2048, 1),
			seqLen(other.getSeqLen())
{
	cout << "rl copy constructor" << endl;
	register_module("fcValue", fcValue);
}

GRUL2RLCombNet& GRUL2RLCombNet::operator=(const GRUL2RLCombNet& other) {
	return *this;
}


GRUL2RLCombNet::GRUL2RLCombNet(const GRUL2RLCombNet&& other) :
			l2Net(std::move(other.l2Net)),
			fcValue(2048, 1),
			seqLen(other.getSeqLen())
{
	cout << "rl move constructor" << endl;
	register_module("fcValue", fcValue);
}

}
