/*
 * MlpBase.cpp
 *
 *  Created on: Jun 18, 2020
 *      Author: zf
 */



#include "gymtest/MlpBase.h"
#include "gymtest/gymnetutils.h"
#include <stdexcept>

using Tensor = torch::Tensor;
using TensorList = torch::TensorList;
using string = std::string;

MlpNet::MlpNet(unsigned iNumInputs, unsigned int iNumActOutput, bool recurrent, unsigned int hiddenSize)
	: actor(nullptr),
	  critic(nullptr),
	  criticLinear(nullptr),
	  actorLinear(nullptr),
	  numInputs(iNumInputs),
	  numActOutput(iNumActOutput)
{
	if (recurrent) {
		throw std::runtime_error("Recurrent not supported");
	}

	actor = torch::nn::Sequential(torch::nn::Linear(numInputs, hiddenSize),
									torch::nn::Functional(torch::tanh),
									torch::nn::Linear(hiddenSize, hiddenSize),
									torch::nn::Functional(torch::tanh));
	critic = torch::nn::Sequential(torch::nn::Linear(numInputs, hiddenSize),
									torch::nn::Functional(torch::tanh),
									torch::nn::Linear(hiddenSize, hiddenSize),
									torch::nn::Functional(torch::tanh));
	criticLinear = torch::nn::Linear(hiddenSize, 1); //only one action output?
	actorLinear = torch::nn::Linear(hiddenSize, numActOutput);

	register_module("actor", actor);
	register_module("critic", critic);
	register_module("criticLinear", criticLinear);
	register_module("actorLinear", actorLinear);

	init_weights(actor->named_parameters(), sqrt(2.0), 0);
	init_weights(critic->named_parameters(), sqrt(2.0), 0);
	init_weights(criticLinear->named_parameters(), sqrt(2.0), 0);
	init_weights(actorLinear->named_parameters(), sqrt(2.0), 0);
}


std::vector<Tensor> MlpNet::forward(Tensor inputs, Tensor hxs, Tensor masks) {
	auto x = inputs;

	auto hiddenCritic = critic->forward(x);
	auto hiddenActor = actor->forward(x);
	auto critic = criticLinear->forward(hiddenCritic);
	auto acts = actorLinear->forward(hiddenActor);
//	auto actsOut = torch::softmax(acts, -1);

	return { critic, acts,
//		actsOut,
		hxs};
}
