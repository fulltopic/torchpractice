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
#include "policy/tenhoupolicy.h"
#include "tenhouclient/logger.h"

class RandomPolicy: public TenhouPolicy {
private:
	float rndRate;
	std::shared_ptr<spdlog::logger> logger;

public:
	RandomPolicy(float rate);
	virtual ~RandomPolicy() = default;

	virtual int getAction(torch::Tensor values, std::vector<int> candidates);
	virtual int getAction(torch::Tensor values, std::vector<int> candidates, std::vector<int> excludes);
	virtual std::vector<int> getTiles4Action(torch::Tensor values, int actionType, std::vector<int> candidates, const int raw);
	virtual void reset();
};


#endif /* INCLUDE_POLICY_RANDOMPOLICY_H_ */
