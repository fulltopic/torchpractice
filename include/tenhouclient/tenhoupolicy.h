/*
 * tenhoupolicy.h
 *
 *  Created on: Apr 19, 2020
 *      Author: zf
 */

#ifndef INCLUDE_TENHOUCLIENT_TENHOUPOLICY_H_
#define INCLUDE_TENHOUCLIENT_TENHOUPOLICY_H_

#include <torch/torch.h>
#include <vector>

//TODO: single instance pattern
class TenhouPolicy {
public:
	TenhouPolicy() = default;
	virtual ~TenhouPolicy() = 0;
	virtual int getAction(torch::Tensor values, std::vector<int> candidates) = 0;
	virtual int getAction(torch::Tensor values, std::vector<int> candidates, std::vector<int> excludes) = 0;
	virtual std::vector<int> getTiles4Action(torch::Tensor values, int actionType, std::vector<int> candidates) = 0;
	virtual void reset() = 0;
};



#endif /* INCLUDE_TENHOUCLIENT_TENHOUPOLICY_H_ */
