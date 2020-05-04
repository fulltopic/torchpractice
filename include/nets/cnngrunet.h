/*
 * cnngrunet.h
 *
 *  Created on: Apr 2, 2020
 *      Author: zf
 */

#ifndef INCLUDE_NETS_CNNGRUNET_H_
#define INCLUDE_NETS_CNNGRUNET_H_

#include <torch/torch.h>
#include <fstream>
#include <iostream>

struct CnnGRUNet: torch::nn::Module {
	torch::nn::Conv2d conv0;
	torch::nn::BatchNorm2d batchNorm0;
	torch::nn::GRU gru0;
	torch::nn::BatchNorm1d batchNorm1;
	torch::nn::Linear fc;
	std::ofstream dataFile;
	const int seqLen;
//	long totalLen;
//	long totalSample;

	CnnGRUNet(int inSeqLen);
	~CnnGRUNet();

//	torch::Tensor forward (torch::Tensor input);
//	torch::Tensor forward(std::vector<torch::Tensor> inputs);
//	torch::Tensor forward(std::vector<torch::Tensor> inputs, bool testLen);
//	torch::Tensor forward(std::vector<torch::Tensor> inputs, const int SeqLen);
	torch::Tensor forward(std::vector<torch::Tensor> inputs, const int seqLen, bool isTrain = true, bool toRecord = false);
	torch::Tensor inputPreprocess(torch::Tensor);
	void initParams();

	static const std::string GetName();

//	Tensor forward (Tensor input)
};




#endif /* INCLUDE_NETS_CNNGRUNET_H_ */
