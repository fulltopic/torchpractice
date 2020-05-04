/*
 * lstm2net.h
 *
 *  Created on: Apr 3, 2020
 *      Author: zf
 */

#ifndef INCLUDE_NETS_LSTM2NET_H_
#define INCLUDE_NETS_LSTM2NET_H_


#include <torch/torch.h>
#include <fstream>
#include <iostream>

struct Lstm2Net: torch::nn::Module {
	torch::nn::LSTM lstm0;
	torch::nn::BatchNorm1d batchNorm0;
	torch::nn::LSTM lstm1;
	torch::nn::BatchNorm1d batchNorm1;
	torch::nn::Linear fc;
	std::ofstream dataFile;
	const int seqLen;
//	long totalLen;
//	long totalSample;

	Lstm2Net(int inSeqLen);
	~Lstm2Net();

//	torch::Tensor forward (torch::Tensor input);
//	torch::Tensor forward(std::vector<torch::Tensor> inputs);
//	torch::Tensor forward(std::vector<torch::Tensor> inputs, bool testLen);
//	torch::Tensor forward(std::vector<torch::Tensor> inputs, const int SeqLen);
	torch::Tensor forward(std::vector<torch::Tensor> inputs, const int seqLen, bool isTrain = true, bool toRecord = false);
	torch::Tensor inputPreprocess(torch::Tensor);

	static const std::string GetName();

//	Tensor forward (Tensor input)
};



#endif /* INCLUDE_NETS_LSTM2NET_H_ */
