/*
 * gymnetutils.cpp
 *
 *  Created on: Jun 18, 2020
 *      Author: zf
 */




#include <torch/torch.h>
#include "gymtest/gymnetutils.h"

//#include "cpprl/model/model_utils.h"
//#include "third_party/doctest.h"

//using namespace torch;
using Tensor = torch::Tensor;

torch::Tensor orthogonal_(Tensor tensor, double gain)
{
    NoGradGuard guard;

//    AT_CHECK(
//        tensor.ndimension() >= 2,
//        "Only tensors with 2 or more dimensions are supported");

    const auto rows = tensor.size(0);
    const auto columns = tensor.numel() / rows;
    auto flattened = torch::randn({rows, columns});

    if (rows < columns)
    {
        flattened.t_();
    }

    // Compute the qr factorization
    Tensor q, r;
    std::tie(q, r) = torch::qr(flattened);
    // Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    auto d = torch::diag(r, 0);
    auto ph = d.sign();
    q *= ph;

    if (rows < columns)
    {
        q.t_();
    }

    tensor.view_as(q).copy_(q);
    tensor.mul_(gain);

    return tensor;
}

torch::Tensor FlattenImpl::forward(torch::Tensor x)
{
    return x.view({x.size(0), -1});
}

void init_weights(torch::OrderedDict<std::string, torch::Tensor> parameters,
                  double weight_gain,
                  double bias_gain)
{
    for (const auto &parameter : parameters)
    {
        if (parameter.value().size(0) != 0)
        {
            if (parameter.key().find("bias") != std::string::npos)
            {
                nn::init::constant_(parameter.value(), bias_gain);
            }
            else if (parameter.key().find("weight") != std::string::npos)
            {
                orthogonal_(parameter.value(), weight_gain);
            }
        }
    }
}

