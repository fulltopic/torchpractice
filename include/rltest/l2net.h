/*
 * l2net.h
 *
 *  Created on: Sep 4, 2020
 *      Author: zf
 */

#ifndef INCLUDE_RLTEST_L2NET_H_
#define INCLUDE_RLTEST_L2NET_H_

#include <torch/torch.h>

#include <string.h>
#include <string>
#include <vector>

namespace rltest {
struct GRUL2Net: torch::nn::Module {
private:
//	const unsigned int numInputs;
//	const unsigned int numActOutput;
	torch::nn::GRU gru0;
	torch::nn::Linear fc;

	const int seqLen;

public:
	GRUL2Net(int inSeqLen);
	GRUL2Net(GRUL2Net& other) = delete;
	GRUL2Net& operator=(GRUL2Net& other) = delete;
	GRUL2Net(GRUL2Net&& other) = delete;
	GRUL2Net& operator=(GRUL2Net&& other) = delete;

	~GRUL2Net() = default;


	void initParams();
	void loadModel(const std::string modelPath);

	torch::Tensor inputPreprocess(torch::Tensor input);
	std::vector<torch::Tensor> forward(std::vector<torch::Tensor> inputs, bool isTrain);
	std::vector<torch::Tensor> forward (std::vector<torch::Tensor> inputs);
	torch::Tensor createHState();
	void reset();

};
}



#endif /* INCLUDE_RLTEST_L2NET_H_ */
