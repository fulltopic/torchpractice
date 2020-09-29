/*
 * datastorequeue.h
 *
 *  Created on: Sep 22, 2020
 *      Author: zf
 */

#ifndef INCLUDE_UTILS_DATASTOREQUEUE_H_
#define INCLUDE_UTILS_DATASTOREQUEUE_H_


#include "utils/dataqueue.hpp"

#include <vector>
#include <torch/torch.h>

#include "storedata.h"

//using StoreDataType = std::vector<std::vector<torch::Tensor>>;
class DataStoreQ {
private:
	DataStoreQ() = default;
	DataStoreQ(const DataStoreQ&) = delete;
	DataStoreQ& operator= (const DataStoreQ&) = delete;

public:
	//TODO: Remove magic data
//	static R1WmQueue<std::vector<std::vector<torch::Tensor>>, 64>&
	static R1WmQueue<std::unique_ptr<StateDataType>, 64>&
	GetDataQ();
};

#endif /* INCLUDE_UTILS_DATASTOREQUEUE_H_ */
