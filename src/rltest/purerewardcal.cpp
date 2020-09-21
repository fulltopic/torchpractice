/*
 * purerewardcal.cpp
 *
 *  Created on: Sep 15, 2020
 *      Author: zf
 */


#include "rltest/purerewardcal.h"

#include "tenhouclient/tenhoustate.h"

using Tensor = torch::Tensor;
using std::vector;

namespace rltest {
PureRewardCal::PureRewardCal(float relayGamma): gamma(relayGamma) {

}

PureRewardCal::~PureRewardCal() {

}

std::vector<torch::Tensor> PureRewardCal::calReturn(const std::vector<torch::Tensor>& datas) {
	Tensor actions = datas[0];
	Tensor labels = datas[1];
	float reward = datas[2].item<float>();
	const int seqLen = actions.size(0); //TODO: ensure sizes

	long* labelPtr = labels.data<long>();

	Tensor returnTensor = torch::zeros({1, seqLen});
	float* returnPtr = returnTensor.data<float>();

	float returnValue = reward;
	for (int i = seqLen - 1; i >= 0; i --) {
		if (labelPtr[i] == ReachAction) {
			returnValue += -10; //TODO: 10 magic value
		}
		returnPtr[i] = returnValue;
		returnValue *= gamma;
	}

	return {returnTensor};
}

}
