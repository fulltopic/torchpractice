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
Tensor Utils::BasicReturnCalc(
		const Tensor rewardTensor, const Tensor labels, const Tensor actions, const int seqLen, float gamma, float penalty) {
	float reward = rewardTensor.item<float>();
	long* labelPtr = labels.data_ptr<long>();
	long* actionPtr = actions.data_ptr<long>();

	Tensor returnTensor = torch::zeros({seqLen, 1});
	float* returnPtr = returnTensor.data_ptr<float>();

	float returnValue = reward;
	for (int i = seqLen - 1; i >= 0; i --) {
		if (labelPtr[i] == ReachAction) {
			returnValue += -10; //TODO: 10 magic value
		}
		if (labelPtr[i] != actionPtr[i]) {
			std::cout << "Mismatch " << labelPtr[i] << " != " << actionPtr[i] << std::endl;
			returnPtr[i] += penalty;
		}
		returnPtr[i] += returnValue;
		returnValue *= gamma;
	}

	return returnTensor;
}

}



