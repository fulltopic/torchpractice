/*
 * randomnet.h
 *
 *  Created on: Apr 17, 2020
 *      Author: zf
 */

#ifndef INCLUDE_TENHOUCLIENT_RANDOMNET_H_
#define INCLUDE_TENHOUCLIENT_RANDOMNET_H_


#include <torch/torch.h>


class RandomNet {
public:
//	~A3CNet() = 0;
	torch::Tensor forward(std::vector<torch::Tensor> inputs);
};


#endif /* INCLUDE_TENHOUCLIENT_RANDOMNET_H_ */
