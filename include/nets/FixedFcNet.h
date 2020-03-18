/*
 * FixedFcNet.h
 *
 *  Created on: Feb 1, 2020
 *      Author: zf
 */

#ifndef INCLUDE_NETS_FIXEDFCNET_H_
#define INCLUDE_NETS_FIXEDFCNET_H_

#include <torch/torch.h>
#include <fstream>
#include <iostream>
#include <string>

#include "FixedKernelNetDef.h"

struct FixedFcNet: torch::nn::Module {
	FixedKernelNetConf def;
//	torch::nn::BatchNorm2d inputBatch;
	torch::nn::Conv2d conv0;
	torch::nn::BatchNorm2d batchNorm0;
	torch::nn::Conv2d conv1;
	torch::nn::BatchNorm2d batchNorm1;
	torch::nn::Conv2d conv2;
	torch::nn::BatchNorm2d batchNorm2;
	torch::nn::Conv2d conv3;
	torch::nn::BatchNorm2d batchNorm3;
//	torch::nn::Conv2d conv4;
//	torch::nn::BatchNorm2d batchNorm4;
//	torch::nn::FeatureDropout conv3_drop;

	torch::nn::Linear fc;
	std::ofstream dataFile;

	FixedFcNet();
	~FixedFcNet();

	torch::Tensor forward(std::vector<torch::Tensor> inputs, const int seqLen, bool isTrain = true, bool toRecord = false);
	torch::Tensor inputPreprocess(torch::Tensor);

	static const std::string GetName();
};



#endif /* INCLUDE_NETS_FIXEDFCNET_H_ */
