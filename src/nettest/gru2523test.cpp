/*
 * gru2523test.cpp
 *
 *  Created on: May 13, 2020
 *      Author: zf
 */

#include "nets/supervisednet/grunet_2523.h"


#include <torch/torch.h>
#include <torch/script.h>

#include "lmdbtools/LmdbReader.h"
#include "lmdbtools/LmdbDataDefs.h"
#include "lmdbtools/Lmdb2RowDataDefs.h"
#include "lmdbtools/LmdbReaderWrapper.h"
//#include "NetDef.h"
#include <matplotlibcpp.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <stdio.h>
#include <bits/stdc++.h>
#include <sys/types.h>
#include <experimental/filesystem>
#include <thread>
#include <cmath>
#include <time.h>
#include <ctime>
#include <chrono>
#include <functional>

using std::string;
using std::vector;
using torch::Tensor;

void testStep() {
	const string modelPath = "/home/zf/workspaces/workspace_cpp/torchpractice/build/models/candidates/GRUNet_1585782523.pt";
	GruNet_2523 net;
	net.loadParams(modelPath);
//	net.loadParams();

	Tensor input = torch::rand({1, 1, 360});
//	Tensor state = torch::zeros({1, 1, 1024});

	Tensor output = net.forward({input});

	std::cout << output.sizes() << std::endl;

	std::cout << output << std::endl;
}

int main() {
	testStep();
}
