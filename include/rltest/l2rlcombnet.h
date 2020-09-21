/*
 * l2rlnet.h
 *
 *  Created on: Sep 9, 2020
 *      Author: zf
 */

#ifndef INCLUDE_RLTEST_L2RLCOMBNET_H_
#define INCLUDE_RLTEST_L2RLCOMBNET_H_

#include "l2net.h"
#include <torch/torch.h>

namespace rltest {
class GRUL2RLCombNet: public
//torch::nn::Module
torch::nn::Cloneable<GRUL2RLCombNet>
{
private:
	GRUL2Net l2Net;
	torch::nn::Linear fcValue;

	const int seqLen;

public:

	GRUL2RLCombNet(int inSeqLen);

//GRUL2RLNet(GRUL2RLNet& other) = delete;
//GRUL2RLNet& operator=(GRUL2RLNet& other) = delete;
//GRUL2RLNet(GRUL2RLNet&& other) = delete;
//GRUL2RLNet& operator=(GRUL2RLNet&& other) = delete;

	GRUL2RLCombNet(const GRUL2RLCombNet& other);
	GRUL2RLCombNet& operator=(const GRUL2RLCombNet& other);
	GRUL2RLCombNet(const GRUL2RLCombNet&& other);
//	GRUL2RLNet& operator=(GRUL2RLNet&& other) = delete;

	~GRUL2RLCombNet() = default;


	void initParams();
	void loadModel(const std::string modelPath);

	torch::Tensor inputPreprocess(torch::Tensor input);
	std::vector<torch::Tensor> forward(std::vector<torch::Tensor> inputs, bool isTrain);
	std::vector<torch::Tensor> forward (std::vector<torch::Tensor> inputs);
	torch::Tensor createHState();
	void reset();

	inline int getSeqLen() const { return seqLen; }

	torch::Tensor getLoss(std::vector<std::vector<torch::Tensor>> inputTensors);
};
}



#endif /* INCLUDE_RLTEST_L2RLCOMBNET_H_ */
