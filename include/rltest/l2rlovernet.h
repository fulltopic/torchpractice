/*
 * l2rlovernet.h
 *
 *  Created on: Sep 14, 2020
 *      Author: zf
 */

#ifndef INCLUDE_RLTEST_L2RLOVERNET_H_
#define INCLUDE_RLTEST_L2RLOVERNET_H_


#include <torch/torch.h>

#include <string.h>
#include <string>
#include <vector>

namespace rltest {
struct GRUL2OverNet: public
//torch::nn::Module
torch::nn::Cloneable<GRUL2OverNet>
{
protected:
//	const unsigned int numInputs;
//	const unsigned int numActOutput;
	torch::nn::GRU gru0;
	torch::nn::Linear fc;

	torch::nn::Linear fcValue;

	int64_t seqLen;


//protected:
//	GRUL2OverNet(const GRUL2OverNet& other);
//	GRUL2OverNet& operator=(const GRUL2OverNet& other);
//	GRUL2OverNet(GRUL2OverNet&& other);
////		GRUL2OverNet& operator=(GRUL2OverNet&& other);
public:
//	friend class torch::nn::Cloneable<GRUL2OverNet>;
//	friend class std::shared_ptr<torch::nn::Module>;
//	friend class std::shared_ptr<GRUL2OverNet>;
	GRUL2OverNet(int inSeqLen);
	GRUL2OverNet(int inSeqLen, bool isL2Model, const std::string modelPath);


	GRUL2OverNet(const GRUL2OverNet& other);
	GRUL2OverNet& operator=(const GRUL2OverNet& other);
	GRUL2OverNet(GRUL2OverNet&& other);
//		GRUL2OverNet& operator=(GRUL2OverNet&& other);


//	GRUL2OverNet(const GRUL2OverNet& other) = delete;
//	GRUL2OverNet& operator=(GRUL2OverNet& other) = delete;
//	GRUL2OverNet(GRUL2OverNet&& other) = delete;
//	GRUL2OverNet& operator=(GRUL2OverNet&& other) = delete;

//	GRUL2OverNet(const GRUL2OverNet& other) = default;
//	GRUL2OverNet& operator=(GRUL2OverNet& other) = default;
//	GRUL2OverNet(GRUL2OverNet&& other) = default;
//	GRUL2OverNet& operator=(GRUL2OverNet&& other) = default;

	~GRUL2OverNet() = default;


	void initParams();
	void loadL2Model(const std::string modelPath);
	void loadModel(const std::string modelPath);

	inline int getSeqLen() const { return seqLen; }

	torch::Tensor inputPreprocess(torch::Tensor input);
	std::vector<torch::Tensor> forward(std::vector<torch::Tensor> inputs, bool isTrain);
	std::vector<torch::Tensor> forward (std::vector<torch::Tensor> inputs);
	torch::Tensor createHState();
	void reset();// override;

	void cloneFrom(const GRUL2OverNet& origNet);
	torch::Tensor getLoss(std::vector<std::vector<torch::Tensor>> inputTensors);

	static std::string GetName();
};
}





#endif /* INCLUDE_RLTEST_L2RLOVERNET_H_ */
