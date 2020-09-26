/*
 * randompolicy.h
 *
 *  Created on: May 9, 2020
 *      Author: zf
 */

#ifndef INCLUDE_POLICY_RANDOMPOLICY_H_
#define INCLUDE_POLICY_RANDOMPOLICY_H_

#include <torch/torch.h>
#include <vector>

#include "../utils/logger.h"
#include "policy/tenhoupolicy.h"

class RandomPolicy: public TenhouPolicy {
private:
	float rndRate;
	std::shared_ptr<spdlog::logger> logger;

public:
	RandomPolicy(float rate);
	virtual ~RandomPolicy();

	virtual int getAction(torch::Tensor values, const std::vector<int>& candidates);
	virtual int getAction(torch::Tensor values, const std::vector<int>& candidates, const std::vector<int>& excludes);
	virtual std::vector<int> getTiles4Action(torch::Tensor values, int actionType, const std::vector<int>& candidates, const int raw);
	virtual void reset();
};


#endif /* INCLUDE_POLICY_RANDOMPOLICY_H_ */
