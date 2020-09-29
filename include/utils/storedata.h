/*
 * storedata.h
 *
 *  Created on: Sep 28, 2020
 *      Author: zf
 */

#ifndef INCLUDE_UTILS_STOREDATA_H_
#define INCLUDE_UTILS_STOREDATA_H_

#include <vector>
#include <torch/torch.h>

enum StorageIndex {
	InputIndex = 0,
	HStateIndex = 1,
	LabelIndex = 2,
	ActionIndex = 3,
	RewardIndex = 4,
};

struct StateDataType {
	using ItemDataType = std::vector<torch::Tensor>;

	ItemDataType trainStates;
	ItemDataType trainHStates;
	ItemDataType trainLabels; //action executed
	ItemDataType trainActions; //action calculated
	float reward;

	StateDataType();
//	: reward(0.0f) {
//	}
	//Others default

	std::vector<std::vector<torch::Tensor>> getData();
//	{
//		return {trainStates, trainHStates, trainLabels, trainActions, {torch::tensor(reward)}};
//	}
};




#endif /* INCLUDE_UTILS_STOREDATA_H_ */
