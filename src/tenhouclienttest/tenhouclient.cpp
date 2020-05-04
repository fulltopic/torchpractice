/*
 * tenhouclient.cpp
 *
 *  Created on: Apr 20, 2020
 *      Author: zf
 */


#include "tenhouclient/filepolicy.h"
#include "tenhouclient/netproxy.h"
#include "tenhouclient/randomnet.h"
#include "tenhouclient/tenhouconsts.h"
#include "tenhouclient/tenhoufsm.h"
#include "tenhouclient/tenhoufsmstate.h"
#include "tenhouclient/tenhoupolicy.h"
#include "tenhouclient/tenhoustate.h"
#include "tenhouclient/logger.h"

#include <thread>
using namespace std;

//void launchFsm(TenhouFsm& fsm) {
//	fsm.rcv();
//}
//
//void testLaunch(int t) {
//	int a = t + 1;
//}

void test(string path) {
	auto logger = Logger::GetLogger();
//	string path = "/home/zf/workspaces/workspace_cpp/torchpractice/src/tenhouclienttest/reachtestlog.txt";
//	string path = "/home/zf/workspaces/workspace_cpp/torchpractice/build/tenhoulogs/tenhoulog38.txt";
	LstmState innState(10, 72, 5);
	FilePolicy policy(path);
	policy.init();
	logger->info("Policy initiated ");

	NetProxy netProxy(innState, policy);
	TenhouFsm fsm(netProxy);
	logger->info("Fsm created ");

	//TODO: Try no ref input argument
	auto f = [&]() {
		logger->debug("In lambda ");
		fsm.rcv();
	};

//	fsm.rcv();
	thread fsmThread(f);
	logger->info("thread created ");
//	thread fsmThread(testLaunch, 3);
	sleep(3);
	fsm.start();
	logger->info("Fsm triggered");

	fsmThread.join();
}

int main(int argc, char** argv) {
	test(string(argv[1]));
}
