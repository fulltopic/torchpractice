/*
 * testtrainworker.cpp
 *
 *  Created on: Sep 14, 2020
 *      Author: zf
 */



#include <string>
#include <iostream>

#include <torch/torch.h>

#include "rltest/l2rlovernet.h"
#include "rltest/trainworker.hpp"
#include "rltest/purerewardcal.h"

#include "policy/randompolicy.h"

#include "lmdbtools/LmdbReaderWrapper.h"
#include "lmdbtools/LmdbReader.h"
#include "lmdbtools/LmdbDataDefs.h"
#include "lmdbtools/Lmdb2RowDataDefs.h"

#include "utils/logger.h"
#include "spdlog/spdlog.h"

using std::cout;
using std::endl;
using std::vector;

using torch::Tensor;

static void startWorker() {
	spdlog::set_level(spdlog::level::debug);
//	Logger::GetLogger()->set_pattern("[%t][%l]: %v");

	int seqLen = 27;
	const std::string modelPath = "/home/zf/workspaces/workspace_cpp/aws/GRU2L2048MaskNet_140000_0.002000_1593719779.pt";
	const std::string overModelPath = "/home/zf/workspaces/workspace_cpp/torchpractice/build/data/GRUL2OverNet_1_32_1603143083.pt";
//	const std::string optModelPath = "/home/zf/workspaces/workspace_cpp/torchpractice/build/data/N5torch5optim7RMSpropE_0_0_1603143083.pt";
	const std::string optModelPath = "";

	//TODO: Constructor could not args may be because that candidates can not be distinguished between normal input and copy constructor
	TrainObj<rltest::GRUL2OverNet, torch::optim::RMSprop, torch::optim::RMSpropOptions> trainObj;
	trainObj.createNets(std::forward<int>(seqLen), std::forward<bool>(true), std::forward<const std::string>(modelPath));
//	trainObj.createNets(std::forward<int>(seqLen), std::forward<bool>(false), std::forward<const std::string>(overModelPath));
//	torch::optim::RMSprop optimizer(net->parameters(), torch::optim::RMSpropOptions(1e-3).eps(1e-8).alpha(0.99));
	trainObj.createOptimizers(optModelPath, std::forward<torch::optim::RMSpropOptions>(torch::optim::RMSpropOptions(1e-3).eps(1e-8).alpha(0.99)));

	rltest::PureRewardCal calc(0.9);
	RandomPolicy policy(1.0f);
	RlTrainWorker<rltest::GRUL2OverNet, torch::optim::RMSprop, torch::optim::RMSpropOptions> worker(calc, policy, trainObj);

	worker.start();
}


int main() {
	startWorker();
}

