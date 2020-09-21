/*
 * l2rlinheritnet.h
 *
 *  Created on: Sep 14, 2020
 *      Author: zf
 */

#ifndef INCLUDE_RLTEST_L2RLINHERITNET_H_
#define INCLUDE_RLTEST_L2RLINHERITNET_H_

#include "l2net.h"
#include <torch/torch.h>

namespace rltest {
class GRUL2RLInheritNet: public
//torch::nn::Module
rltest::GRUL2Net,
torch::nn::Cloneable<GRUL2RLInheritNet>
{
private:
	torch::nn::Linear fcValue;
//	const int seqLen;

public:

	GRUL2RLInheritNet(int inSeqLen);

//GRUL2RLNet(GRUL2RLNet& other) = delete;
//GRUL2RLNet& operator=(GRUL2RLNet& other) = delete;
//GRUL2RLNet(GRUL2RLNet&& other) = delete;
//GRUL2RLNet& operator=(GRUL2RLNet&& other) = delete;

	GRUL2RLInheritNet(const GRUL2RLInheritNet& other);
	GRUL2RLInheritNet& operator=(const GRUL2RLInheritNet& other);
	GRUL2RLInheritNet(const GRUL2RLInheritNet&& other);
//	GRUL2RLNet& operator=(GRUL2RLNet&& other) = delete;

	~GRUL2RLInheritNet() = default;


	void initParams();
	void loadModel(const std::string modelPath);

	torch::Tensor inputPreprocess(torch::Tensor input);
	std::vector<torch::Tensor> forward(std::vector<torch::Tensor> inputs, bool isTrain);
	std::vector<torch::Tensor> forward (std::vector<torch::Tensor> inputs);
	torch::Tensor createHState();
	void reset();

//	inline int getSeqLen() const { return seqLen; }

	torch::Tensor getLoss(std::vector<std::vector<torch::Tensor>> inputTensors);
};
}



#endif /* INCLUDE_RLTEST_L2RLINHERITNET_H_ */
