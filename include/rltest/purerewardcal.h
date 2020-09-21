/*
 * purerewardcal.h
 *
 *  Created on: Sep 15, 2020
 *      Author: zf
 */

#ifndef INCLUDE_RLTEST_PUREREWARDCAL_H_
#define INCLUDE_RLTEST_PUREREWARDCAL_H_

#include "returncal.h"

#include <vector>
#include <torch/torch.h>

namespace rltest {

class PureRewardCal: public ReturnCalculator {
private:
	const float gamma;

public:
	PureRewardCal(float relayGamma);
	//TODO: Other constructors

	virtual std::vector<torch::Tensor> calReturn(const std::vector<torch::Tensor>& datas
			) ;

	virtual ~PureRewardCal();
};

}


#endif /* INCLUDE_RLTEST_PUREREWARDCAL_H_ */
