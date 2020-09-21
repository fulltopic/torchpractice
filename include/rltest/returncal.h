/*
 * returncal.h
 *
 *  Created on: Sep 7, 2020
 *      Author: zf
 */

#ifndef INCLUDE_RLTEST_RETURNCAL_H_
#define INCLUDE_RLTEST_RETURNCAL_H_

#include <torch/torch.h>
#include <vector>

namespace rltest {
class ReturnCalculator {
public:
	virtual std::vector<torch::Tensor> calReturn(const std::vector<torch::Tensor>& datas
			) = 0;

	virtual ~ReturnCalculator();
};

}

#endif /* INCLUDE_RLTEST_RETURNCAL_H_ */
