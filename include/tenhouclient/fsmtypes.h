/*
 * fsmtypes.h
 *
 *  Created on: May 13, 2020
 *      Author: zf
 */

#ifndef INCLUDE_TENHOUCLIENT_FSMTYPES_H_
#define INCLUDE_TENHOUCLIENT_FSMTYPES_H_

#include <string>

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



#endif /* INCLUDE_TENHOUCLIENT_FSMTYPES_H_ */
