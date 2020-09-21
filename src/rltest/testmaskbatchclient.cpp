/*
 * testmaskbatchclient.cpp
 *
 *  Created on: Sep 3, 2020
 *      Author: zf
 */


#include <vector>
#include <string>
#include <iostream>

#include <boost/asio.hpp>
#include <boost/thread/thread.hpp>
#include <boost/bind/bind.hpp>

#include <torch/torch.h>

#include "lmdbtools/LmdbReaderWrapper.h"
#include "lmdbtools/LmdbReader.h"
#include "lmdbtools/LmdbDataDefs.h"
#include "lmdbtools/Lmdb2RowDataDefs.h"

#include "rltest/maskbatchnet.h"
#include "rltest/l2net.h"
#include "rltest/rltestsetting.h"

#include "policy/randompolicy.h"
#include "tenhouclient/netproxy.hpp"

#include "tenhouclient/tenhouconsts.h"
#include "tenhouclient/tenhoufsm.h"
#include "tenhouclient/tenhoufsmstate.h"
#include "tenhouclient/tenhoustate.h"
#include "tenhouclient/asiotenhoufsm.hpp"

using Tensor = torch::Tensor;
using std::cout;
using std::endl;
using std::vector;
//using rltest::GRUMaskNet;


namespace {
const std::string wsPath = "/home/zf/";
//std::ofstream dataFile(wsPath + "/workspaces/workspace_cpp/torchpractice/build/errorstats.txt");
const std::string batchModelPath = "/home/zf/workspaces/workspace_cpp/aws/GRUMaskBatch2048Net_140000_0.002000_1594188556.pt";
const std::string l2ModelPath = "/home/zf/workspaces/workspace_cpp/aws/GRU2L2048MaskNet_140000_0.002000_1593719779.pt";
const int seqLen = 27;
const int batchSize = 128;
//const std::string name = "ID5F706D6D-2WBML2Pe"; //testrl0
const std::string name = "ID715C4B99-dSNcQnGe"; //testrl02
}

static void testBatchnorm(const std::string modelPath) {
	BaseState innState(72, 5);
	RandomPolicy policy(1.0);

	auto netPtr = std::shared_ptr<rltest::GRUMaskNet>(new rltest::GRUMaskNet(seqLen));
	netPtr->loadModel(modelPath);
	auto netProxy = std::shared_ptr<NetProxy<rltest::GRUMaskNet>>(
			new NetProxy<rltest::GRUMaskNet>(rltest::RlSetting::Names[0], netPtr, innState, policy));
	boost::asio::io_context io;

	auto pointer = asiotenhoufsm<rltest::GRUMaskNet>::Create(io, netProxy,
			rltest::RlSetting::ServerIp, rltest::RlSetting::ServerPort, name);
	pointer->start();

	io.run();
}

static void testL2(const std::string modelPath) {
	BaseState innState(72, 5);
	RandomPolicy policy(1.0);

	auto netPtr = std::shared_ptr<rltest::GRUL2Net>(new rltest::GRUL2Net(seqLen));
	netPtr->loadModel(modelPath);
	auto netProxy = std::shared_ptr<NetProxy<rltest::GRUL2Net>>(
			new NetProxy<rltest::GRUL2Net>(rltest::RlSetting::Names[0], netPtr, innState, policy));
	boost::asio::io_context io;

	auto pointer = asiotenhoufsm<rltest::GRUL2Net>::Create(io, netProxy,
			rltest::RlSetting::ServerIp, rltest::RlSetting::ServerPort, name);
	pointer->start();

	io.run();
}

int main(int argc, char** argv) {
//	testBatchnorm(batchModelPath);
	testL2(l2ModelPath);
}
