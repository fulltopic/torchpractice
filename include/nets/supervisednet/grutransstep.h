/*
 * grutransstep.h
 *
 *  Created on: May 19, 2020
 *      Author: zf
 */

#ifndef INCLUDE_NETS_SUPERVISEDNET_GRUTRANSSTEP_H_
#define INCLUDE_NETS_SUPERVISEDNET_GRUTRANSSTEP_H_



#include <torch/torch.h>
#include <fstream>
#include <iostream>
#include <vector>

struct GRUTransStepNet: torch::nn::Module {
	torch::nn::GRU gru0;
	torch::nn::BatchNorm1d batchNorm0;
	torch::nn::Linear fc;
	std::vector<torch::nn::BatchNorm1d> stepBatchNorms;
//	const int batchSize;
//	long totalLen;
//	long totalSample;
	static thread_local int step;
	static thread_local bool resetState;
	static thread_local torch::Tensor state;

	GRUTransStepNet();
	~GRUTransStepNet() = default;

//	torch::Tensor forward (torch::Tensor input);
//	torch::Tensor forward(std::vector<torch::Tensor> inputs);
//	torch::Tensor forward(std::vector<torch::Tensor> inputs, bool testLen);
//	torch::Tensor forward(std::vector<torch::Tensor> inputs, const int SeqLen);
	void reset();
	torch::Tensor forward(std::vector<torch::Tensor> inputs, bool isBatch);
	torch::Tensor forward(std::vector<torch::Tensor> inputs);
	torch::Tensor inputPreprocess(torch::Tensor);
	void initParams();
	void loadParams(const std::string modelPath);

	static const std::string GetName();

private:
	torch::Tensor forwardBatch(std::vector<torch::Tensor>& inputs);
	torch::Tensor forwardStep(std::vector<torch::Tensor>& inputs);

//	Tensor forward (Tensor input)
};



#endif /* INCLUDE_NETS_SUPERVISEDNET_GRUTRANSSTEP_H_ */
