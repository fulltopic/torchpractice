/*
 * tenhoufsmstate.cpp
 *
 *  Created on: Apr 14, 2020
 *      Author: zf
 */


#include "tenhouclient/tenhoufsmstate.h"

#include "../../include/utils/logger.h"
#include "tenhouclient/tenhoufsm.h"

#include "tenhouclient/tenhoumsggenerator.h"
#include "tenhouclient/tenhoumsgparser.h"
#include "tenhouclient/tenhoumsgutils.h"


using namespace std;

using G=TenhouMsgGenerator;
using P=TenhouMsgParser;
using U=TenhouMsgUtils;
/****************** Start State ************************/
//const int StartState::getType() {
//	return StateType::StartState;
//}

#define ConstructState(StateName) \
	StateName::StateName(TenhouFsm& f): TenhouFsmState(f) {}

#define DestructState(StateName) \
		StateName::~StateName() {}

TenhouFsmState::TenhouFsmState(TenhouFsm& f): fsm(f) {}
TenhouFsmState::~TenhouFsmState() {}

HeloState::HeloState(TenhouFsm& f): TenhouFsmState(f) {}
ConstructState(StartState);
ConstructState(AuthState);
ConstructState(JoinState);
ConstructState(RejoinState);
ConstructState(ReadyState);
ConstructState(GamingState);
ConstructState(GameEndState);
ConstructState(SceneEndState);
ConstructState(ErrorState);

DestructState(HeloState);
DestructState(StartState);
DestructState(AuthState);
DestructState(JoinState);
DestructState(RejoinState);
DestructState(ReadyState);
DestructState(GamingState);
DestructState(GameEndState);
DestructState(SceneEndState);
DestructState(ErrorState);

StateReturnType StartState::rcv(string msg) {
	if (msg.find(TenhouFsm::StartFsm) != string::npos) {
		return {G::GenHeloMsg("NoName"), StateType::HeloStateType};
	} if (msg.find("TIMEOUT") != string::npos) {
		return {G::GenHeloMsg("NoName"), StateType::HeloStateType};
	}
	else {
		return {msg, StateType::ErrorStateType};
	}
}

/***************** Helo State ***************************/
StateReturnType HeloState::rcv(string msg) {
	Logger::GetLogger()->debug("Helo received msg {}", msg);
	if (msg.find("HELO") != string::npos) {
		Logger::GetLogger()->debug("Helo parse helo message ");
		auto parts = P::ParseHeloReply(msg);
		for (int i = 0; i < parts.size(); i ++) {
			Logger::GetLogger()->debug("Part {}: {}", i, parts[i]);
		}
		string authMsg = G::GenAuthReply(parts);
		Logger::GetLogger()->debug("Get auth msg {}", authMsg);
		string pxrMsg = G::GenPxrMsg();
		Logger::GetLogger()->debug("Get pxr msg {} ", pxrMsg);

		return {authMsg + StateReturnType::SplitToken + pxrMsg, StateType::AuthStateType};
	} else {
		return {msg, StateType::ErrorStateType};
	}
}

/***************** Auth State **************************/
StateReturnType AuthState::rcv(string msg) {
	if (msg.find("LN") != string::npos) {
		return {G::GenJoinMsg(), StateType::JoinStateType};
	} else if (msg.find("REJOIN") != string::npos) {
		return {G::GenRejoinMsg(msg), StateType::JoinStateType};
	} else {
		return {msg, StateType::ErrorStateType};
	}
}

StateReturnType JoinState::rcv(string msg) {
	if (msg.find("GO") != string::npos) {
		return {StateReturnType::Nothing, StateType::JoinStateType };
	} else if (msg.find("LN") != string::npos) {
//		return {StateReturnType::Nothing, StateType::JoinStateType };
		return {G::GenPxrMsg(), StateType::JoinStateType };
	} else if (msg.find("UN") != string::npos) {
		return {StateReturnType::Nothing, StateType::JoinStateType };
	}else if (msg.find("TAIKYOKU") != string::npos) {
		return {G::GenGoMsg() + StateReturnType::SplitToken + G::GenNextReadyMsg()
			+ StateReturnType::SplitToken + G::GenNextReadyMsg(),
			StateType::ReadyStateType };
	}else if (msg.find("REJOIN") != string::npos) {
		return { G::GenRejoinMsg(msg), StateType::JoinStateType};
	} else {
		return { msg, StateType::ErrorStateType };
	}
}


//TODO: Timer
StateReturnType ReadyState::rcv(string msg) {
	if (msg.find("INIT") != string::npos) {
		fsm.reset();
		string rc = fsm.process(msg);
		return { rc, StateType::GamingStateType };
	} else if (msg.find("PROF") != string::npos) {
//		return { G::GenByeMsg(), StateType::SceneEndStateType};
		return { StateReturnType::Nothing, StateType::SceneEndStateType};
	} else if (P::IsGameEnd(msg)) {
		string rc = fsm.process(msg);
//		return { rc, StateType::GameEndStateType };
		return {G::GenNextReadyMsg(), StateType::ReadyStateType};
	}else {
		return { msg, StateType::ErrorStateType };
	}
}

StateReturnType GameEndState::rcv(string msg) {
	//TODO: To replace by boost and timer
	sleep(1);
	if (msg.find("PROF") != string::npos) {
		return { StateReturnType::Nothing, StateType::SceneEndStateType};
	}
//	else {
//		return {G::GenNextReadyMsg(), StateType::ReadyStateType};
//	}
	else if (msg.find("TIMEOUT") != string::npos) {
//		return { G::GenNextReadyMsg(), StateType::ReadyStateType};
		return { G::GenNoopMsg(), StateType::ReadyStateType };
	} else if (P::IsGameEnd(msg)) {
		string rc = fsm.process(msg);
//		return { rc, StateType::GameEndStateType };
		return {G::GenNextReadyMsg(), StateType::GameEndStateType};
	}
	else {
		return { msg, StateType::ErrorStateType };
	}
}

//TODO: send BYE in timeout case
StateReturnType ErrorState::rcv(string msg) {
	cout << "Received unexpected msg: " << msg << endl;
	exit(-1);
	return { StateReturnType::Nothing, StateType::ErrorStateType};
}

StateReturnType GamingState::rcv(string msg) {
	if (msg.find("PROF") != string::npos) {
		//Wait for game end message
		return {StateReturnType::Nothing, StateType::SceneEndStateType };
	}

	if (!U::IsGameMsg(msg)) {
		cout << "Received unexpected msg: " << msg << endl;
		return { StateReturnType::Nothing, StateType::GamingStateType};
	}

	//TODO: Timer msg
	bool isTerminal = U::IsTerminalMsg(msg);
	string rc = fsm.process(msg);

	if (isTerminal) {
//		return { rc, StateType::GameEndStateType };
		return {G::GenNextReadyMsg(), StateType::ReadyStateType};
	} else {
		return { rc, StateType::GamingStateType };
	}
}

StateReturnType SceneEndState::rcv(string msg) {
	if (msg.find("TIMEOUT") != string::npos) {
		return { G::GenByeMsg(), StateType::StartStateType };
	} else {
		if (U::IsTerminalMsg(msg)) {
			//TODO: To process message for net. Done
			fsm.process(msg);
			return { G::GenNextReadyMsg(), StateType::SceneEndStateType };
		} else {
			cout << "Received unexpected message in SceneEndState: " << msg << endl;
			return { G::GenByeMsg(), StateType::HeloStateType};
		}
	}
}

StateReturnType RejoinState::rcv(string msg) {
	if (msg.find("GO") != string::npos) {
		return {G::GenGoMsg(), StateType::ReadyStateType };
	} else if (msg.find("UN") != string::npos) {
		return { StateReturnType::Nothing, StateType::RejoinStateType };
	} else {
		return { msg, StateType::ErrorStateType };
	}
}
