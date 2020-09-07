/*
 * testmjclient.cpp
 *
 *  Created on: May 9, 2020
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
#include <memory>

#include "../../include/tenhouclient/asiotenhoufsm.hpp"
#include "tenhouclient/netproxy.hpp"
#include "policy/dirpolicy.h"
#include "policy/filepolicy.h"
#include "policy/tenhoupolicy.h"
#include "policy/randompolicy.h"

#include "nets/grustep.h"
#include "nets/supervisednet/grunet_2523.h"
#include "nets/supervisednet/grunet_5334.h"
#include "nets/supervisednet/grutransstep.h"

void test() {
//	std::string modelPath = "/home/zf/workspaces/workspace_cpp/torchpractice/build/models/candidates/GRUNet_1585782523.pt";
//	std::string modelPath = "/home/zf/workspaces/workspace_cpp/torchpractice/build/models/candidates/GRUStep_1589645334.pt";
	std::string modelPath = "/home/zf/workspaces/workspace_cpp/torchpractice/build/GRUNet_1589879376.pt";


	auto logger = Logger::GetLogger();

	BaseState innState(72, 5);
	RandomPolicy policy(0.1);
//	policy.init();
	logger->info("Policy initiated ");

//	auto netPtr = std::shared_ptr<GruNet_5334>(new GruNet_5334());
//	netPtr->loadParams(modelPath);
//	NetProxy<GruNet_5334> netProxy(netPtr, innState, policy);

	auto netPtr = std::shared_ptr<GRUTransStepNet>(new GRUTransStepNet());
	netPtr->loadParams(modelPath);
	NetProxy<GRUTransStepNet> netProxy(netPtr, innState, policy);
//	NetProxy<GRUStepNet> netProxy(std::shared_ptr<GRUStepNet>(new GRUStepNet()), innState, policy);
	boost::asio::io_context io;

	auto pointer = asiotenhoufsm<GRUTransStepNet>::Create(io, netProxy, "NoName");
	pointer->start();

	io.run();
//	boost::thread t(boost::bind(&boost::asio::io_context::run, &io));
//
//	t.join();
}

int main(int argc, char** argv) {
	test();
}



