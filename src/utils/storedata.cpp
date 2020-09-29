/*
 * storedata.cpp
 *
 *  Created on: Sep 28, 2020
 *      Author: zf
 */




#include "utils/storedata.h"
#include <vector>
#include <torch/torch.h>

StateDataType::StateDataType(): reward(0.0f) {}

std::vector<std::vector<torch::Tensor>> StateDataType::getData() {
	return {trainStates, trainHStates, trainLabels, trainActions, {torch::tensor(reward)}};
}
