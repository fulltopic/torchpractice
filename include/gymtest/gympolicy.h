/*
 * gympolicy.h
 *
 *  Created on: Jun 18, 2020
 *      Author: zf
 */

#ifndef INCLUDE_GYMTEST_GYMPOLICY_H_
#define INCLUDE_GYMTEST_GYMPOLICY_H_


#include <torch/torch.h>

class GymPolicy {
public:
	GymPolicy() = default;
	~GymPolicy() = default;

	torch::Tensor getAct(torch::Tensor actOutput);
};


#endif /* INCLUDE_GYMTEST_GYMPOLICY_H_ */
