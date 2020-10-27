/*
 * gaecalc.h
 *
 *  Created on: Oct 5, 2020
 *      Author: zf
 */

#ifndef INCLUDE_RLTEST_GAECALC_H_
#define INCLUDE_RLTEST_GAECALC_H_

#include <torch/torch.h>

namespace rltest{
class GAECal {
private:
	const float gamma;
	const float lambda;

public:
	GAECal(float gammaF, float lambdaF);
	torch::Tensor calc(torch::Tensor values, torch::Tensor reward);
	torch::Tensor calcAdv (torch::Tensor values, torch::Tensor reward);
};
}



#endif /* INCLUDE_RLTEST_GAECALC_H_ */
