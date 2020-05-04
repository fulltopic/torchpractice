/*
 * a3cnet.cpp
 *
 *  Created on: Apr 20, 2020
 *      Author: zf
 */



#include "nets/a3cnet.h"

using namespace torch;
using namespace std;

Tensor A3CNet::forward(vector<Tensor> inputs) {
	return torch::rand({42});
}
