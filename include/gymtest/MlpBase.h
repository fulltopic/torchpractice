/*
 * MlpBase.h
 *
 *  Created on: Jun 18, 2020
 *      Author: zf
 */

#ifndef INCLUDE_GYMTEST_MLPBASE_H_
#define INCLUDE_GYMTEST_MLPBASE_H_



#include <torch/torch.h>
#include <vector>

struct MlpNet: torch::nn::Module {
private:
	torch::nn::Sequential actor;
	torch::nn::Sequential critic;
	torch::nn::Linear criticLinear;
	torch::nn::Linear actorLinear;
	const unsigned int numInputs;
	const unsigned int numActOutput;

public:
	MlpNet(unsigned int iNumInputs,
			unsigned int iNumActOutput,
				bool recurrent = false,
				unsigned int hiddenSize = 64);

	std::vector<torch::Tensor> forward(torch::Tensor inputs,
										torch::Tensor hxs,
										torch::Tensor masks);

	inline unsigned int getNumInputs() const {
		return numInputs;
	}
};


#endif /* INCLUDE_GYMTEST_MLPBASE_H_ */
