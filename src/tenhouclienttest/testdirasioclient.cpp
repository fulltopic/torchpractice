/*
 * testdirasioclient.cpp
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

#include "tenhouclient/asiotenhoufsm.hpp"
#include "tenhouclient/netproxy.hpp"
#include "policy/dirpolicy.h"
#include "policy/filepolicy.h"
#include "policy/tenhoupolicy.h"

#include "nets/grustep.h"


void test(std::string path) {
	auto logger = Logger::GetLogger();

	LstmState innState(72, 5);
	DirPolicy policy(path);
	policy.init();
	logger->info("Policy initiated ");

//	NetProxy<RandomNet> netProxy(std::shared_ptr<RandomNet>(new RandomNet()), innState, policy);
	NetProxy<GRUStepNet> netProxy(std::shared_ptr<GRUStepNet>(new GRUStepNet()), innState, policy);
	boost::asio::io_context io;

	auto pointer = asiotenhoufsm<GRUStepNet>::Create(io, netProxy);
	pointer->start();

	io.run();
//	boost::thread t(boost::bind(&boost::asio::io_context::run, &io));
//
//	t.join();
}

int main(int argc, char** argv) {
	test(argv[1]);
}
