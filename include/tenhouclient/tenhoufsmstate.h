/*
 * tenhoufsmstate.h
 *
 *  Created on: Apr 13, 2020
 *      Author: zf
 */

#ifndef INCLUDE_NETS_TENHOUFSMSTATE_H_
#define INCLUDE_NETS_TENHOUFSMSTATE_H_

#include "tenhoufsm.h"

#define StateImp(StateName) \
	class StateName: public TenhouFsmState { \
	public: \
		StateName(TenhouFsm& f);	\
		virtual ~StateName();	\
		virtual StateReturnType rcv(std::string msg);	\
		virtual inline const int getType() {	\
			return StateType::StateName##Type;	\
		} \
	};

class HeloState: public TenhouFsmState {
public:
	HeloState(TenhouFsm& f);
	virtual ~HeloState();

	virtual StateReturnType rcv(std::string msg);
//	virtual int next();
	virtual inline const int getType() { return HeloStateType; };
};

StateImp(StartState)

StateImp(AuthState);

StateImp(JoinState);

StateImp(RejoinState);

StateImp(ReadyState);

StateImp(GamingState);

StateImp(GameEndState);

StateImp(SceneEndState);

StateImp(ErrorState);




#endif /* INCLUDE_NETS_TENHOUFSMSTATE_H_ */
