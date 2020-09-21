/*
 * rltestutils.cpp
 *
 *  Created on: Sep 8, 2020
 *      Author: zf
 */

#include <torch/torch.h>
#include "rltest/rltestutils.h"

#include "tenhouclient/tenhoustate.h"

using Tensor = torch::Tensor;
using std::vector;

namespace rltest{
bool Utils::CompTensorBySeqLen (const torch::Tensor& t0, const torch::Tensor& t1) {
	return t0.size(0) > t1.size(0);
}

//TODO: Normalize rewards
Tensor Utils::BasicReturnCalc(const Tensor rewardTensor, const Tensor labels, const int seqLen, float gamma) {
	float reward = rewardTensor.item<float>();
	long* labelPtr = labels.data<long>();

	Tensor returnTensor = torch::zeros({seqLen, 1});
	float* returnPtr = returnTensor.data<float>();

	float returnValue = reward;
	for (int i = seqLen - 1; i >= 0; i --) {
		if (labelPtr[i] == ReachAction) {
			returnValue += -10; //TODO: 10 magic value
		}
		returnPtr[i] = returnValue;
		returnValue *= gamma;
	}

	return returnTensor;
}
}


