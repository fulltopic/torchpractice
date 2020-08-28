/*
 * meanstd.cpp
 *
 *  Created on: Jun 27, 2020
 *      Author: zf
 */




#include <torch/torch.h>

#include "gymtest/meanstd.h"

RunningMeanStdImpl::RunningMeanStdImpl(int size)
    : count(register_buffer("count", torch::full({1}, 1e-4, torch::kFloat))),
      mean(register_buffer("mean", torch::zeros({size}))),
      variance(register_buffer("variance", torch::ones({size}))) {}

RunningMeanStdImpl::RunningMeanStdImpl(std::vector<float> means, std::vector<float> variances)
    : count(register_buffer("count", torch::full({1}, 1e-4, torch::kFloat))),
      mean(register_buffer("mean", torch::from_blob(means.data(), {static_cast<long>(means.size())})
                                       .clone())),
      variance(register_buffer("variance", torch::from_blob(variances.data(), {static_cast<long>(variances.size())})
                                               .clone())) {}

void RunningMeanStdImpl::update(torch::Tensor observation)
{
    observation = observation.reshape({-1, mean.size(0)});
    auto batch_mean = observation.mean(0);
    auto batch_var = observation.var(0, false, false);
    auto batch_count = observation.size(0);

    update_from_moments(batch_mean, batch_var, batch_count);
}

void RunningMeanStdImpl::update_from_moments(torch::Tensor batch_mean,
                                             torch::Tensor batch_var,
                                             int batch_count)
{
    auto delta = batch_mean - mean;
    auto total_count = count + batch_count;

    mean.copy_(mean + delta * batch_count / total_count);
    auto m_a = variance * count;
    auto m_b = batch_var * batch_count;
    auto m2 = m_a + m_b + torch::pow(delta, 2) * count * batch_count / total_count;
    variance.copy_(m2 / total_count);
    count.copy_(total_count);
}


