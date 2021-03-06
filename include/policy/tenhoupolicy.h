/*
 * tenhoupolicy.h
 *
 *  Created on: Apr 19, 2020
 *      Author: zf
 */

#ifndef INCLUDE_POLICY_TENHOUPOLICY_H_
#define INCLUDE_POLICY_TENHOUPOLICY_H_

#include <torch/torch.h>
#include <vector>

//TODO: single instance pattern
class TenhouPolicy {
public:
	TenhouPolicy() = default;
	virtual ~TenhouPolicy() = 0;
	virtual int getAction(torch::Tensor values, const std::vector<int>& candidates) = 0;
	virtual int getAction(torch::Tensor values, const std::vector<int>& candidates, const std::vector<int>& excludes) = 0;
	virtual std::vector<int> getTiles4Action(torch::Tensor values, int actionType, const std::vector<int>& candidates, const int raw) = 0;
	virtual void reset() = 0;
};



#endif /* INCLUDE_POLICY_TENHOUPOLICY_H_ */
