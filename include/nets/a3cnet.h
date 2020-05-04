/*
 * a3cnet.h
 *
 *  Created on: Apr 13, 2020
 *      Author: zf
 */

#ifndef INCLUDE_NETS_A3CNET_H_
#define INCLUDE_NETS_A3CNET_H_

#include <torch/torch.h>


class A3CNet {
public:
//	~A3CNet() = 0;
	torch::Tensor forward(std::vector<torch::Tensor> inputs);
};



#endif /* INCLUDE_NETS_A3CNET_H_ */
