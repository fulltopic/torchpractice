/*
 * testdbreader4cnn.cpp
 *
 *  Created on: Mar 7, 2020
 *      Author: zf
 */


#include "lmdbtools/LmdbReader.h"
#include "lmdbtools/LmdbDataDefs.h"

#include <torch/torch.h>

#include <iostream>
#include <string>

void testCnnDim() {
	const std::string dbPath = "/home/zf/workspaces/res/dbs/lmdbcpptest";
//	const std::string dbType = "lmdb";

	LmdbSceneReader<LmdbDataDefs> reader(dbPath);

	torch::Tensor data;
	torch::Tensor label;

	std::tie(data, label) = reader.next();

	std::cout << data.sizes() << std::endl;
	std::cout << label.sizes() << std::endl;
}

int main() {
	testCnnDim();

	return 0;
}

