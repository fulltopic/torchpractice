#ifndef INCLUDE_LSTMNET_H_
#define INCLUDE_LSTMNET_H_

#include <torch/torch.h>
#include <fstream>
#include <iostream>

struct LstmNet: torch::nn::Module {
	torch::nn::Conv2d conv0;
	torch::nn::Conv2d conv1;
	torch::nn::LSTM lstm0;
	torch::nn::Linear fc;
	std::ofstream dataFile;
	long totalLen;
	long totalSample;

	LstmNet();
	~LstmNet();

	torch::Tensor forward (torch::Tensor input);
	torch::Tensor forward(std::vector<torch::Tensor> inputs);
	torch::Tensor forward(std::vector<torch::Tensor> inputs, bool testLen);
	torch::Tensor forward(std::vector<torch::Tensor> inputs, const int SeqLen);
	torch::Tensor forward(std::vector<torch::Tensor> inputs, const int seqLen, bool isTrain = true, bool toRecord = false);
//	Tensor forward (Tensor input)
};

#endif
