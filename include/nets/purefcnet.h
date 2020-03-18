/*
 * purefcnet.h
 *
 *  Created on: Feb 16, 2020
 *      Author: zf
 */

#ifndef INCLUDE_NETS_PUREFCNET_H_
#define INCLUDE_NETS_PUREFCNET_H_

#include <torch/torch.h>
#include <fstream>
#include <iostream>

struct PureFcNet: torch::nn::Module {
	torch::nn::Conv2d conv0;
	torch::nn::BatchNorm2d batchNorm0;
	torch::nn::Linear fc;
	std::ofstream dataFile;

	PureFcNet();
	~PureFcNet();

	torch::Tensor forward(std::vector<torch::Tensor> inputs, const int seqLen, bool isTrain = true, bool toRecord = false);
};



#endif /* INCLUDE_NETS_PUREFCNET_H_ */
