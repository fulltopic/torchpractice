/*
 * room.h
 *
 *  Created on: Oct 27, 2020
 *      Author: zf
 */

#ifndef INCLUDE_SELFSERVER_ROOM_H_
#define INCLUDE_SELFSERVER_ROOM_H_

#include <ctime>
#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <set>
#include <random>
#include <mutex>
#include <stdlib.h>     /* srand, rand */
#include <time.h>

//#include <torch/torch.h>
//#include <boost/asio.hpp>
#include <boost/enable_shared_from_this.hpp>

#include "clientconn.h"
#include "playerstate.h"
#include "tilepatternutils.h"

struct IndReq {
	int clientIndex;
	int fromWho;
	int raw;
	MeldType indType;
	bool received;
	bool accepted;
	bool rsped;
	int rspSeq;
	std::vector<int> raws;
};

class Room;
//class WaitingObj {
//private:
//	std::vector<IndReq> reqs;
////	std::vector<int> rsps;
//
//	std::mutex m;
//	int rspSeq;
//
//	void resetNoLock();
//public:
//	WaitingObj();
//	~WaitingObj() = default;
//
//	void req(int index, int fromWho, int indType, int raw);
//	bool receive(int index);
//	void accept(int index, int rspType, std::vector<int> raws);
//	void process (int index);
//
//	void reset();
//
//	bool received(int index);
//	bool accepted(int index);
//	bool processed(int index);
//	bool reqed(int index);
//	bool reqed();
//	std::vector<int> getActInfo(int index);
//	inline int getFromWho(int index) { return reqs[index].fromWho;}
////	int getRaw(int index);
////	int getIndType(int index);
////	int getFromWho(int index);
//	bool isAgari(int index);
//	bool isReach(int index);
//	bool allRspRcvedNoLock();
//	bool allRspRcved();
//	int getRspIndex();
//	int getNextDistIndex();
//	inline int getIndType(int index) { return reqs[index].indType;}
//	inline std::vector<int> getMeldRaws(int index) { return reqs[index].raws; }
//
//	void printReqInfo(int index);
//	void printReqsInfo();
//
//	friend class Room;
//};

class Room: public std::enable_shared_from_this<Room> {
private:
	class WaitingObj {
	private:
		std::vector<IndReq> reqs = std::vector<IndReq>(TilePatternUtils::PlayerNum);
	//	std::vector<int> rsps;

		std::mutex m;
		int rspSeq = 0;

		void resetNoLock();
	public:
		WaitingObj();
		~WaitingObj() = default;

		void req(int index, int fromWho, MeldType indType, int raw);
		bool receive(int index);
		void accept(int index, MeldType rspType, std::vector<int> raws);

		void reset();

		bool received(int index);
		bool accepted(int index);
		bool processed(int index);
		bool reqed(int index);
		bool reqed();
		std::tuple<int, MeldType, int> getActInfo(int index);
		inline int getFromWho(int index) { return reqs[index].fromWho;}
	//	int getRaw(int index);
	//	int getIndType(int index);
	//	int getFromWho(int index);
		bool isAgari(int index);
		bool isReach(int index);
		bool allRspRcvedNoLock();
		bool allRspRcved();
		int getRspIndex();
		int getNextDistIndex();
		inline MeldType getIndType(int index) { return reqs[index].indType;}
		inline std::vector<int> getMeldRaws(int index) { return reqs[index].raws; }

		void printReqInfo(int index);
		void printReqsInfo();

//		friend class Room;
	};

	std::vector<std::shared_ptr<ClientConn>> clients
		= std::vector<std::shared_ptr<ClientConn>> (TilePatternUtils::PlayerNum);

	//random
	std::random_device rd;
	std::mt19937 gen;// = std::mt19937(rd());
	std::vector<int> wall = std::vector<int>(136, 0);
//	std::vector<int> m

	std::vector<PlayerState> states;
	std::vector<int> doras;
	std::vector<int> doraSteps;
	std::vector<int> tens
		= std::vector<int>(TilePatternUtils::PlayerNum, TilePatternUtils::InitTen);;

	WaitingObj wo;
	std::map<int, int> reachRaws;

	std::mutex bar;
	std::mutex closeMutex;
	volatile int allReady = 0;
	volatile int allBye = 0;
	volatile int allGo = 0;

	volatile bool working = true;
	volatile int roomState;

	int oyaIndex = -1;
	int tileIndex = 0;
	int nextClientIndex = -1;

//	torch::Tensor rndTensor;
//	long* wallData = nullptr;

//	void clientReady(int clientIndex);

	void gameInit();
	void gameEnd();
	void sceneInit();
	int decideNextOya();
	int getAbsIndex(const int index, const int clientIndex) const;
	int getRelaIndex(const int index, const int clientIndex) const;

	void sendInitMsg(int clientIndex);

	bool processOrphanCase();
	bool processAbortCase();
	bool processAuthMsg(int clientIndex, std::string& msg);
	void processInitMsg(int clientIndex, std::string& msg);
	void processDropMsg(int cientIndex, std::string& msg);
	void processIndMsg(int clientIndex, std::string& msg);
	void processReachMsg(int clientIndex, std::string& msg);
	void processSceneEndMsg(int clientIndex, std::string& msg);

	void processRyu();
	void processAgari(int who, int fromWho);
	void processBye(int clientIndex);

	void distRaw (int clientIndex);

protected:
	explicit Room(uint32_t iSeq);
public:
	const uint32_t seq;

	enum {
		AllReady = 0x0f,
	};
	~Room();

	static std::shared_ptr<Room> Create(uint32_t iSeq);

	void addClient(int index, std::shared_ptr<ClientConn> client);

	void processMsg(int clientIndex, std::string& msg); //Should be thread_safe
	void processNetErr(int clientIndex);

	inline bool isWorking() { return working; }
	inline int getClientNum() { return clients.size(); }
	inline bool isSceneReady() { return (allReady == AllReady); }
};


#endif /* INCLUDE_SELFSERVER_ROOM_H_ */
