/*
 * rnnmasknet.h
 *
 *  Created on: May 23, 2020
 *      Author: zf
 */

#ifndef INCLUDE_NETS_RNNMASKNET_H_
#define INCLUDE_NETS_RNNMASKNET_H_


#include <torch/torch.h>
#include <fstream>
#include <iostream>

struct GRUMaskNet: torch::nn::Module {
	torch::nn::GRU gru0;
	torch::nn::BatchNorm1d batchNorm0;
	torch::nn::Linear fc;
//	std::ofstream dataFile;
	const int seqLen;
//	long totalLen;
//	long totalSample;

	GRUMaskNet(int inSeqLen);
	~GRUMaskNet() = default;

//	torch::Tensor forward (torch::Tensor input);
//	torch::Tensor forward(std::vector<torch::Tensor> inputs);
//	torch::Tensor forward(std::vector<torch::Tensor> inputs, bool testLen);
//	torch::Tensor forward(std::vector<torch::Tensor> inputs, const int SeqLen);
	torch::Tensor forward(std::vector<torch::Tensor> inputs, bool isBatch);
	torch::Tensor forward(std::vector<torch::Tensor> inputs, const int seqLen, bool isTrain, bool toRecord);
	torch::Tensor inputPreprocess(torch::Tensor);
	void initParams();

	static const std::string GetName();

//	Tensor forward (Tensor input)
};



#endif /* INCLUDE_NETS_RNNMASKNET_H_ */
