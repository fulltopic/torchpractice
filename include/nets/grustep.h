/*
 * grustep.h
 *
 *  Created on: May 14, 2020
 *      Author: zf
 */

#ifndef INCLUDE_NETS_GRUSTEP_H_
#define INCLUDE_NETS_GRUSTEP_H_


#include <torch/torch.h>
#include <fstream>
#include <iostream>

struct GRUStepNet: torch::nn::Module {
	torch::nn::GRU gru0;
	torch::nn::BatchNorm1d batchNorm0;
	torch::nn::Linear fc;
//	const int batchSize;
//	long totalLen;
//	long totalSample;
	static thread_local bool resetState;
	static thread_local torch::Tensor state;

	GRUStepNet();
	~GRUStepNet() = default;

//	torch::Tensor forward (torch::Tensor input);
//	torch::Tensor forward(std::vector<torch::Tensor> inputs);
//	torch::Tensor forward(std::vector<torch::Tensor> inputs, bool testLen);
//	torch::Tensor forward(std::vector<torch::Tensor> inputs, const int SeqLen);
	void reset();
	std::vector<torch::Tensor> forward(std::vector<torch::Tensor> inputs, bool isTrain);
	std::vector<torch::Tensor> forward(std::vector<torch::Tensor> inputs);
	torch::Tensor inputPreprocess(torch::Tensor);
	void initParams();
	void loadParams(const std::string modelPath);

	static const std::string GetName();

//	Tensor forward (Tensor input)
};




#endif /* INCLUDE_NETS_GRUSTEP_H_ */
