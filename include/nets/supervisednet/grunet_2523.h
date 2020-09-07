/*
 * grunet_2523.h
 *
 *  Created on: May 13, 2020
 *      Author: zf
 */

#ifndef INCLUDE_NETS_SUPERVISEDNET_GRUNET_2523_H_
#define INCLUDE_NETS_SUPERVISEDNET_GRUNET_2523_H_

#include <torch/torch.h>

#include <vector>
#include <iostream>
#include <string>

#include "tenhouclient/logger.h"

class GruNet_2523: torch::nn::Module {
public:
	torch::nn::GRU gru0;
	torch::nn::BatchNorm1d batchNorm0;
	torch::nn::Linear fc;

//	std::string modelPath;

	static thread_local bool resetState;
	static thread_local torch::Tensor state;

//	std::shared_ptr<spdlog::logger> logger;

	GruNet_2523();
	~GruNet_2523() = default;

//	void initParams();
//	void loadParams();

	torch::Tensor inputPreprocess(torch::Tensor);
	std::vector<torch::Tensor> forward(std::vector<torch::Tensor> inputs, bool isTrain);
	std::vector<torch::Tensor> forward(std::vector<torch::Tensor> inputs);
	void loadParams(const std::string modelPath);
	void reset();

	static const std::string GetName();
};



#endif /* INCLUDE_NETS_SUPERVISEDNET_GRUNET_2523_H_ */
