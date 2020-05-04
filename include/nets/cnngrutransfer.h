/*
 * cnngrutransfer.h
 *
 *  Created on: Apr 7, 2020
 *      Author: zf
 */

#ifndef INCLUDE_NETS_CNNGRUTRANSFER_H_
#define INCLUDE_NETS_CNNGRUTRANSFER_H_

#include "torch/torch.h"
#include <vector>

#include <torch/torch.h>
#include <fstream>
#include <iostream>
#include <string>

struct CNNGRUTransfer: torch::nn::Module {
	torch::nn::Conv2d conv0;
	torch::nn::BatchNorm2d batchNorm0;
	torch::nn::Conv2d conv1;
	torch::nn::BatchNorm2d batchNorm1;
	torch::nn::Conv2d conv2;
	torch::nn::BatchNorm2d batchNorm2;
	torch::nn::GRU gru0;
	torch::nn::BatchNorm1d batchNormGru0;
	torch::nn::Linear fc0;
	torch::nn::BatchNorm1d fcBatchNorm0;
	torch::nn::Linear fc1;

	std::ofstream dataFile;
	const int seqLen;

	CNNGRUTransfer(const int inSeq, const torch::nn::Module& transferredNet);
	~CNNGRUTransfer();

	void initParams();
	void loadParams(const torch::nn::Module& transferredNet);
	void setTrain(bool isTrain);
	torch::Tensor forward(std::vector<torch::Tensor> inputs, const int seqLen, bool isTrain = true, bool toRecord = false);
	torch::Tensor inputPreprocess(torch::Tensor);

	static const std::string GetName();
};




#endif /* INCLUDE_NETS_CNNGRUTRANSFER_H_ */
