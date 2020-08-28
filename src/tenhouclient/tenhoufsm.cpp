/*
 * tenhoufsm.cpp
 *
 *  Created on: Apr 14, 2020
 *      Author: zf
 */



#include "tenhouclient/tenhoufsmstate.h"
#include "tenhouclient/tenhoufsm.h"

#include "tenhouclient/tenhoumsggenerator.h"
#include "tenhouclient/tenhoumsgparser.h"
#include "tenhouclient/tenhoumsgutils.h"
#include "tenhouclient/randomnet.h"

#include "tenhouclient/tenhoustate.h"

#include <boost/algorithm/string.hpp>
#include "tenhouclient/netproxy.hpp"

using namespace std;


const string TenhouFsm::StartFsm = "STARTFSM";

TenhouFsm::~TenhouFsm() {
	for (auto ite = states.begin(); ite != states.end(); ite ++) {
		TenhouFsmState* state = ite->second;
		ite->second = nullptr;
		delete state;
	}
}

//TODO: Try unique_ptr
TenhouFsm::TenhouFsm(NetProxy<GRUStepNet>& iNet):
		states {
	{StartStateType, new StartState(*this) },
	{HeloStateType, new HeloState(*this) },
	{AuthStateType, new AuthState(*this) },
	{JoinStateType, new JoinState(*this) },
	{RejoinStateType, new RejoinState(*this) },
	{ReadyStateType, new ReadyState(*this) },
	{GamingStateType, new GamingState(*this) },
	{GameEndStateType, new GameEndState(*this) },
	{SceneEndStateType, new SceneEndState(*this) },
	{ErrorStateType, new ErrorState(*this) }
	},
	curState(states[StartStateType]),
	net(iNet),
	logger(Logger::GetLogger())//, testNet(iNet)
{
		conn.connServer();
//	LstmState testState(10, 1, 1);
//	RandomPolicy testP(0.1);
//	NetProxy test(testState, testP);
//	test.processMsg("");
//
//	iNet.processMsg("");
//	net.processMsg("");
//	testNet.processMsg("");
}

void TenhouFsm::rcvMsg(string msg){
	StateReturnType next = curState->rcv(msg);
	string sendMsg = next.msg;
	curState = states[next.nextState];
	logger->debug("Get fsm result {}, {}", next.nextState, next.msg);

	if (next.msg.length() > 0) {
		if (sendMsg.find(StateReturnType::SplitToken) != string::npos) {
			vector<string> items;
			boost::split(items, sendMsg,
					boost::is_any_of(StateReturnType::SplitToken), boost::token_compress_on);
			logger->debug("To sent splitted msg {}", items.size());
			for (int i = 0; i < items.size(); i ++) {
				conn.connSend(items[i]);
				logger->debug("Sent {}", items[i]);
				sleep(1);
			}
		} else {
			logger->debug("Sent {}", sendMsg);
			conn.connSend(sendMsg);
			sleep(1);
		}
	}
}

void TenhouFsm::rcv() {
	//TODO: ASIO
	{
		logger->info("Waiting to start");
		std::unique_lock<std::mutex> lock(mtx);
		waitObj.wait(lock);

		auto rc = curState->rcv(StartFsm);
		curState = states[rc.nextState];
		conn.connSend(rc.msg);
	}
	while (true) {
		//TODO:
		string msg;
//		if (curState->getType() == StateType::GameEndStateType) {
//			msg = "TIMER";
//		} else {
//			msg = conn.connRcv();
//		}
		msg = conn.connRcv();

//		string msg = conn.connRcv();
		logger->debug("Received message {}", msg);
		//TODO: Some message to be dropped
		if (!TenhouMsgParser::IsValidMsg(msg)) {
			logger->warn("Pass invalid msg: {}", msg);
			continue;
		}

		int lastIndex = 0;
		int index = msg.find(">", lastIndex);

		if ((index == string::npos) && msg.find("TIMEOUT") != string::npos) {
			logger->debug("To process msg {}", msg);
			rcvMsg(msg);
			continue;
		}

		while (index != string::npos) {
			string subMsg = msg.substr(lastIndex, (index - lastIndex + 1));
			boost::trim(subMsg);
			logger->debug("To process msg {}", subMsg);

			rcvMsg(subMsg);

//			StateReturnType next = curState->rcv(subMsg);
//			string sendMsg = next.msg;
//			curState = states[next.nextState];
//			logger->debug("Get fsm result {}, {}", next.nextState, next.msg);
//
//			if (next.msg.length() > 0) {
//				if (sendMsg.find(StateReturnType::SplitToken) != string::npos) {
//					vector<string> items;
//					boost::split(items, sendMsg,
//							boost::is_any_of(StateReturnType::SplitToken), boost::token_compress_on);
//					logger->debug("To sent splitted msg {}", items.size());
//					for (int i = 0; i < items.size(); i ++) {
//						conn.connSend(items[i]);
//						logger->debug("Sent {}", items[i]);
//						sleep(1);
//					}
//				} else {
//					logger->debug("Sent {}", sendMsg);
//					conn.connSend(sendMsg);
//					sleep(1);
//				}
//			}

			lastIndex = index + 1;
			index = msg.find(">", lastIndex);
		}
//		StateReturnType next = curState->rcv(msg);
//		string sendMsg = next.msg;
//		curState = states[next.nextState];
//		logger->debug("Get fsm result {}, {}", next.nextState, next.msg);
//
//		if (next.msg.length() > 0) {
//			if (sendMsg.find(StateReturnType::SplitToken) != string::npos) {
//				vector<string> items;
//				boost::split(items, sendMsg,
//						boost::is_any_of(StateReturnType::SplitToken), boost::token_compress_on);
//				logger->debug("To sent splitted msg {}", items.size());
//				for (int i = 0; i < items.size(); i ++) {
//					conn.connSend(items[i]);
//					logger->debug("Sent {}", items[i]);
//					sleep(1);
//				}
//			} else {
//				logger->debug("Sent {}", sendMsg);
//				conn.connSend(sendMsg);
//				sleep(1);
//			}
//		}
	}
}

void TenhouFsm::reset() {
	curState = states[StartStateType];
}

string TenhouFsm::process(string msg) {
	return net.processMsg(msg);
}

void TenhouFsm::start() {
	logger->info("Start fsm");
	cout << "To start fsm " << endl;
	std::unique_lock<std::mutex> lock(mtx);
	waitObj.notify_all();
}
