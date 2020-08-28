/*
 * tenhoufsm.h
 *
 *  Created on: Apr 13, 2020
 *      Author: zf
 */

#ifndef INCLUDE_TENHOUCLIENT_TENHOUFSM_H_
#define INCLUDE_TENHOUCLIENT_TENHOUFSM_H_

#include <vector>
#include <map>
#include <string>

#include <mutex>
#include <condition_variable>

#include "tenhouconn.h"
#include "logger.h"
#include "netproxy.hpp"
#include "randomnet.h"
#include "fsmtypes.h"

#include "nets/grustep.h"


class TenhouFsm;
class TenhouFsmState {
protected:
	TenhouFsm& fsm;

public:
	TenhouFsmState(TenhouFsm& f);
	virtual ~TenhouFsmState() = 0;

	virtual StateReturnType rcv(std::string msg) = 0;
	virtual const int getType() = 0;
};

//class TenhouStateStorage {
//private:
//	std::map<int, TenhouState> states;
//
//public:
//	TenhouStateStorage() = default;
//	TenhouStateStorage(TenhouStateStorage& c) = delete;
//	TenhouStateStorage& operator=(TenhouStateStorage& c) = delete;
//
//	TenhouState& getState(int stateType);
//};

class TenhouFsm {
private:
	std::map<StateType, TenhouFsmState*> states;
	TenhouFsmState* curState;
	NetProxy<GRUStepNet>& net;
	TenhouTcpConn conn;
	std::mutex mtx;
	std::condition_variable waitObj;
	std::shared_ptr<spdlog::logger> logger;

	void rcvMsg(std::string msg);
//	NetProxy testNet;

public:
	static const std::string StartFsm;

	TenhouFsm(NetProxy<GRUStepNet>& iNet);
	TenhouFsm(TenhouFsm&) = delete;
//	TenhouFsm(TenhouFsm&) = default;
	TenhouFsm& operator=(TenhouFsm&) = delete;
	~TenhouFsm();


	void start();
	std::string process(std::string);
	void rcv();
	void reset();
};



#endif /* INCLUDE_TENHOUCLIENT_TENHOUFSM_H_ */
