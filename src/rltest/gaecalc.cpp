/*
 * gaecalc.cpp
 *
 *  Created on: Oct 5, 2020
 *      Author: zf
 */



#include <vector>
#include <torch/torch.h>
#include "rltest/gaecalc.h"
#include "rltest/rltestsetting.h"

#include <iostream>

namespace rltest{
//lambda = 0.95, gamma = 0.99 by default
GAECal::GAECal(float gammaF, float lambdaF): gamma(gammaF), lambda(lambdaF) {

}

torch::Tensor GAECal::calc(torch::Tensor values, torch::Tensor reward) {
	const int seqLen = values.size(0);
	torch::Tensor returnValue = torch::zeros({seqLen, 1});
	reward = torch::clamp(reward, -RlSetting::RewardClip, RlSetting::RewardClip);
	float rewardValue = reward.item<float>();
	std::cout << "-------------------------> Reward: " << rewardValue << std::endl;

	auto valuePtr = values.data_ptr<float>();
	auto returnPtr = returnValue.data_ptr<float>();

	float gae = 0;
	float lastValue = rewardValue;
	for (int i = seqLen - 1; i >= 0; i --) {
		float delta = 0;
		if (i == seqLen - 1) {
			delta = rewardValue - valuePtr[i]; //mask[i] = 0
		} else {
			delta = gamma * lastValue - valuePtr[i]; //rewards[i] = 0
		}

		lastValue = valuePtr[i];
		gae = delta + gamma * lambda * gae;
		returnPtr[i] = gae + valuePtr[i];
	}

	std::cout << "--------------------------------> returnValue: " << std::endl << returnValue << std::endl;
	return returnValue;
}

torch::Tensor GAECal::calcAdv(torch::Tensor values, torch::Tensor returnTensor) {
	torch::Tensor adv = returnTensor - values; //{seqLen, 1}

//	torch::Tensor meanTensor = adv.mean();
//	float mean = meanTensor.item<float>();
//
//	torch::Tensor stdTensor = adv.std();
//	float stdValue = stdTensor.item<float>();
//
//	adv = (adv - mean) / (stdValue + 1e-10);
//
//	std::cout << "-------------------------> mean = " << mean << " std = " << stdValue << std::endl;
	std::cout << "adv: " << std::endl;
	std::cout << adv << std::endl;
	return adv;
}
}
