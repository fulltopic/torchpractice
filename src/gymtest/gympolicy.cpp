/*
 * gympolicy.cpp
 *
 *  Created on: Jun 18, 2020
 *      Author: zf
 */



#include "gymtest/gympolicy.h"

#include <iostream>

using Tensor = torch::Tensor;

Tensor GymPolicy::getAct(Tensor actOutput) {
	float* dataPtr = actOutput.data<float>();
	for (int i = 0; i < actOutput.numel(); i ++) {
		if (dataPtr[i] < 0) {
			std::cout << actOutput << std::endl;
			break;
		}
	}
	return torch::multinomial(actOutput, 1, true);
}
