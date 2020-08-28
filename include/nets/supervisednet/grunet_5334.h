/*
 * grunet_5334.h
 *
 *  Created on: May 18, 2020
 *      Author: zf
 */

#ifndef INCLUDE_NETS_SUPERVISEDNET_GRUNET_5334_H_
#define INCLUDE_NETS_SUPERVISEDNET_GRUNET_5334_H_

#include <torch/torch.h>

#include <vector>
#include <iostream>
#include <string>

#include "tenhouclient/logger.h"

class GruNet_5334: torch::nn::Module {
public:
	torch::nn::GRU gru0;
	torch::nn::BatchNorm1d batchNorm0;
	torch::nn::Linear fc;

//	std::string modelPath;

	static thread_local bool resetState;
	static thread_local torch::Tensor state;

//	std::shared_ptr<spdlog::logger> logger;

	GruNet_5334();
	~GruNet_5334() = default;

//	void initParams();
//	void loadParams();

	torch::Tensor inputPreprocess(torch::Tensor);
	torch::Tensor forward(std::vector<torch::Tensor> inputs, bool isTrain);
	torch::Tensor forward(std::vector<torch::Tensor> inputs);
	void loadParams(const std::string modelPath);
	void reset();

	static const std::string GetName();
};






#endif /* INCLUDE_NETS_SUPERVISEDNET_GRUNET_5334_H_ */
