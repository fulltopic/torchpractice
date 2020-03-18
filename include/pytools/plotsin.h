/*
 * plotsin.h
 *
 *  Created on: Nov 14, 2019
 *      Author: zf
 */

#ifndef INCLUDE_PLOTSIN_H_
#define INCLUDE_PLOTSIN_H_

#include <torch/torch.h>

void plot(std::vector<torch::Tensor> datas, torch::Tensor xAxis,
		std::vector<std::string> colors, std::string fileName);



#endif /* INCLUDE_PLOTSIN_H_ */
