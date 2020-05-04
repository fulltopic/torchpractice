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

#include "netproxy.h"
#include "tenhouconn.h"
#include "logger.h"

enum StateType {
	StartStateType,
	HeloStateType,
	AuthStateType,
	JoinStateType,
	RejoinStateType,
	ReadyStateType,

//	NextReadyStateType,
	GamingStateType,

	GameEndStateType,
	SceneEndStateType,

	ErrorStateType,
};

//using StateReturnType = std::pair<std::string, int>;
struct StateReturnType {
	static const std::string SplitToken;
	static const std::string Nothing;

	std::string msg;
	StateType nextState;
};

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
	NetProxy& net;
	TenhouTcpConn conn;
	std::mutex mtx;
	std::condition_variable waitObj;
	std::shared_ptr<spdlog::logger> logger;

	void rcvMsg(std::string msg);
//	NetProxy testNet;

public:
	static const std::string StartFsm;

	TenhouFsm(NetProxy& iNet);
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
