/*
 * fcnet.h
 *
 *  Created on: Jan 22, 2020
 *      Author: zf
 */

#ifndef INCLUDE_FCNET_H_
#define INCLUDE_FCNET_H_

#include <torch/torch.h>
#include <fstream>
#include <iostream>

struct FcNet: torch::nn::Module {
	torch::nn::Conv2d conv0;
	torch::nn::BatchNorm2d batchNorm0;
	torch::nn::Conv2d conv1;
	torch::nn::BatchNorm2d batchNorm1;
	torch::nn::Linear fc;
	std::ofstream dataFile;

	FcNet();
	~FcNet();

//	torch::Tensor forward (torch::Tensor input);
//	torch::Tensor forward(std::vector<torch::Tensor> inputs);
//	torch::Tensor forward(std::vector<torch::Tensor> inputs, bool testLen);
//	torch::Tensor forward(std::vector<torch::Tensor> inputs, const int SeqLen);
	torch::Tensor forward(std::vector<torch::Tensor> inputs, const int seqLen, bool isTrain = true, bool toRecord = false);
};



#endif /* INCLUDE_FCNET_H_ */
