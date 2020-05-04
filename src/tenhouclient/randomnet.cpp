/*
 * randomnet.cpp
 *
 *  Created on: Apr 20, 2020
 *      Author: zf
 */





#include "tenhouclient/randomnet.h"

using namespace torch;
using namespace std;

Tensor RandomNet::forward(vector<Tensor> inputs) {
	return torch::rand({42});
}
