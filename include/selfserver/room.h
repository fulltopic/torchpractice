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

#include <torch/torch.h>
//#include <boost/asio.hpp>
#include <boost/enable_shared_from_this.hpp>
//#include <boost/smart_ptr/shared_ptr.hpp>

#include "clientconn.h"
#include "playerstate.h"

struct IndReq {
	int clientIndex;
	int fromWho;
	int raw;
	int indType;
	bool received;
	bool accepted;
	bool rsped;
	int rspSeq;
	std::vector<int> raws;
};

class Room;
class WaitingObj {
private:
	std::vector<IndReq> reqs;
//	std::vector<int> rsps;

	std::mutex m;
	int rspSeq;

	void resetNoLock();
public:
	WaitingObj();
	~WaitingObj() = default;

	void req(int index, int fromWho, int indType, int raw);
	void receive(int index);
	void accept(int index, int rspType, std::vector<int>&& raws);
	void process (int index);

	void reset();

	bool received(int index);
	bool accepted(int index);
	bool processed(int index);
	bool reqed(int index);
	bool reqed();
	std::vector<int> getActInfo(int index);
	inline int getFromWho(int index) { return reqs[index].fromWho;}
//	int getRaw(int index);
//	int getIndType(int index);
//	int getFromWho(int index);
	bool isAgari(int index);
	bool isReach(int index);
	bool allRspRcved();
	int getRspIndex();
	int getNextDistIndex();
	inline int getIndType(int index) { return reqs[index].indType;}
	inline std::vector<int> getMeldRaws(int index) { return reqs[index].raws; }

	void printReqInfo(int index);
	void printReqsInfo();

	friend class Room;
};

class Room: public std::enable_shared_from_this<Room> {
private:
	std::vector<std::shared_ptr<ClientConn>> clients;
	std::vector<int> wall;
//	std::vector<int> m

	std::vector<PlayerState> states;
//	std::vector<std::vector<int>> playerTiles;
//	std::vector<std::vector<int>> drops;
//	std::vector<std::vector<int>> deals;
	std::vector<int> doras;
	std::vector<int> doraSteps;
//	std::vector<bool> reached;
	std::vector<int> tens;
//	std::vector<IndReq> waitingReqs;
//	std::vector<int> waitingRsps;

	WaitingObj wo;
	std::map<int, int> reachRaws;

	std::mutex bar;
	volatile int allReady;
	volatile int allBye;
	volatile int allGo;

	volatile bool working;

	int oyaIndex;
	int tileIndex;
	int nextClientIndex;

//	std::mt19937 gen;
	torch::Tensor rndTensor;
	long* wallData;

	Room(uint32_t iSeq);

//	void clientReady(int clientIndex);

	void gameInit();
	void gameEnd();
	void sceneInit();
	int decideNextOya();

	void sendInitMsg(int clientIndex);

	bool processOrphanCase();
	bool processAbortCase();
	bool processInitMsg(int clientIndex, std::string& msg);
	void processDropMsg(int cientIndex, std::string& msg);
	void processIndMsg(int clientIndex, std::string& msg);
	void processReachMsg(int clientIndex, std::string& msg);

	void processRyu();
	void processAgari(int who, int fromWho);
	void processBye(int clientIndex);

	void distRaw (int clientIndex);

public:
	const uint32_t seq;

	enum {
		AllReady = 0x0f,
	};
	~Room() = default;

	static std::shared_ptr<Room> Create(uint32_t iSeq);

	void addClient(int index, std::shared_ptr<ClientConn> client);

	void processMsg(int clientIndex, std::string msg); //Should be thread_safe

	inline bool isWorking() { return working; }
	inline int getClientNum() { return clients.size(); }
	inline bool isSceneReady() { return (allReady == AllReady); }
};


#endif /* INCLUDE_SELFSERVER_ROOM_H_ */
