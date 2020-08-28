/*
 * gymnetutils.h
 *
 *  Created on: Jun 18, 2020
 *      Author: zf
 */

#ifndef INCLUDE_GYMTEST_GYMNETUTILS_H_
#define INCLUDE_GYMTEST_GYMNETUTILS_H_


#pragma once

#include <vector>

#include <torch/torch.h>

using namespace torch;

struct FlattenImpl : nn::Module
{
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(Flatten);

void init_weights(torch::OrderedDict<std::string, torch::Tensor> parameters,
                  double weight_gain,
                  double bias_gain);


#endif /* INCLUDE_GYMTEST_GYMNETUTILS_H_ */
