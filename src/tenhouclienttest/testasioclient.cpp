/*
 * testasioclient.cpp
 *
 *  Created on: May 8, 2020
 *      Author: zf
 */



#include "tenhouclient/randomnet.h"
#include "tenhouclient/tenhouconsts.h"
#include "tenhouclient/tenhoufsm.h"
#include "tenhouclient/tenhoufsmstate.h"
#include "tenhouclient/tenhoustate.h"
#include "tenhouclient/logger.h"

#include <boost/asio.hpp>
#include <boost/thread/thread.hpp>
#include <boost/bind/bind.hpp>
#include <thread>

#include "tenhouclient/asiotenhoufsm.hpp"
#include "policy/filepolicy.h"
#include "policy/tenhoupolicy.h"
#include "tenhouclient/netproxy.hpp"

#include "nets/supervisednet/grunet_2523.h"

void test(std::string path) {
	std::string modelPath = "/home/zf/workspaces/workspace_cpp/torchpractice/build/models/candidates/GRUNet_1585782523.pt";

	auto logger = Logger::GetLogger();

	LstmState innState(72, 5);
	FilePolicy policy(path);
	policy.init();
	logger->info("Policy initiated ");

	auto netPtr = std::shared_ptr<GruNet_2523>(new GruNet_2523());
	netPtr->loadParams(modelPath);
	NetProxy<GruNet_2523> netProxy(netPtr, innState, policy);
//	NetProxy<GRUStepNet> netProxy(std::shared_ptr<GRUStepNet>(new GRUStepNet()), innState, policy);
	boost::asio::io_context io;

	auto pointer = asiotenhoufsm<GruNet_2523>::Create(io, netProxy);
	pointer->start();

	io.run();
//	boost::thread t(boost::bind(&boost::asio::io_context::run, &io));
//
//	t.join();
}

int main(int argc, char** argv) {
	test(argv[1]);
}
