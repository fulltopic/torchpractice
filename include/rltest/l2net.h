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
struct GRUL2Net: public
//torch::nn::Module
torch::nn::Cloneable<GRUL2Net>
{
protected:
//	const unsigned int numInputs;
//	const unsigned int numActOutput;
	torch::nn::GRU gru0;
	torch::nn::Linear fc;

	int seqLen;

//protected:
//	GRUL2Net(const GRUL2Net& other);
//	GRUL2Net& operator=(const GRUL2Net& other);
//	GRUL2Net(GRUL2Net&& other);
////		GRUL2Net& operator=(GRUL2Net&& other);
public:
//	friend class torch::nn::Cloneable<GRUL2Net>;
//	friend class std::shared_ptr<torch::nn::Module>;
//	friend class std::shared_ptr<GRUL2Net>;

	GRUL2Net(const GRUL2Net& other);
	GRUL2Net& operator=(const GRUL2Net& other);
	GRUL2Net(GRUL2Net&& other);
//		GRUL2Net& operator=(GRUL2Net&& other);

	GRUL2Net(int inSeqLen);

//	GRUL2Net(const GRUL2Net& other) = delete;
//	GRUL2Net& operator=(GRUL2Net& other) = delete;
//	GRUL2Net(GRUL2Net&& other) = delete;
//	GRUL2Net& operator=(GRUL2Net&& other) = delete;

//	GRUL2Net(const GRUL2Net& other) = default;
//	GRUL2Net& operator=(GRUL2Net& other) = default;
//	GRUL2Net(GRUL2Net&& other) = default;
//	GRUL2Net& operator=(GRUL2Net&& other) = default;

	~GRUL2Net() = default;


	void initParams();
	void loadModel(const std::string modelPath);

	inline int getSeqLen() const { return seqLen; }

	torch::Tensor inputPreprocess(torch::Tensor input);
	std::vector<torch::Tensor> forward(std::vector<torch::Tensor> inputs, bool isTrain);
	std::vector<torch::Tensor> forward (std::vector<torch::Tensor> inputs);
	torch::Tensor createHState();
	void reset();// override;

	void cloneFrom(const GRUL2Net& origNet);
	torch::Tensor getLoss(std::vector<std::vector<torch::Tensor>> inputTensors);

};
}



#endif /* INCLUDE_RLTEST_L2NET_H_ */
