/*
 * datastorequeue.cpp
 *
 *  Created on: Sep 22, 2020
 *      Author: zf
 */


#include "utils/datastorequeue.h"


R1WmQueue<std::vector<std::vector<torch::Tensor>>, 64>&
	DataStoreQ::GetDataQ() {
	static R1WmQueue<std::vector<std::vector<torch::Tensor>>, 64> queue;

	return queue;
}

