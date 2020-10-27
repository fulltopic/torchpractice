/*
 * rltestutils.h
 *
 *  Created on: Sep 8, 2020
 *      Author: zf
 */

#ifndef INCLUDE_RLTEST_RLTESTUTILS_H_
#define INCLUDE_RLTEST_RLTESTUTILS_H_

#include <torch/torch.h>

namespace rltest{
class Utils {
public:
	static bool CompTensorBySeqLen (const torch::Tensor& t0, const torch::Tensor& t1);

	static torch::Tensor BasicReturnCalc(
			const torch::Tensor reward, const torch::Tensor labels, const torch::Tensor actions, const int seqLen, float gamma, float penalty);

};
}



#endif /* INCLUDE_RLTEST_RLTESTUTILS_H_ */
