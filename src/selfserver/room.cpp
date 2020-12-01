/*
 * room.cpp
 *
 *  Created on: Oct 28, 2020
 *      Author: zf
 */


#include "selfserver/clientconn.h"
#include "selfserver/room.h"

#include "tenhouclient/fsmtypes.h"
#include "tenhouclient/tenhoumsgparser.h"
#include "utils/logger.h"

#include <string>
#include <iostream>
#include <chrono>
#include <random>
#include <algorithm>
#include <map>
#include <set>
#include <mutex>
#include <stdlib.h>     /* srand, rand */
#include <time.h>
#include <limits>

#include <boost/algorithm/string.hpp>

using S = std::string;
using PU = TilePatternUtils;
using std::vector;
using P = TenhouMsgParser;

//TODO: Which part of memory does the anonymous variables reside?
namespace {
	std::map<int, S> dropMsgHead {
		{0, "D"},
		{1, "E"},
		{2, "F"},
		{3, "G"}
	};

	std::map<int, S> acceptMsgHead {
		{0, "T"},
		{1, "U"},
		{2, "V"},
		{3, "W"},
	};

	enum RoomState {
		AuthState = 0,
		InitState = 1,
		DropState = 2,
		IndState = 3,
		ReachState = 4,
		SceneEndState = 5,
		InvalidState = 8,
	};

	auto logger = Logger::GetLogger();
}

Room::WaitingObj::WaitingObj()
{
	for (int i = 0; i < PU::PlayerNum; i ++) {
		reqs[i].clientIndex = i;
	}
	resetNoLock();
}

void Room::WaitingObj::resetNoLock() {
	for (int i = 0; i < PU::PlayerNum; i ++) {
		reqs[i].indType = MeldType::Invalid;
		reqs[i].received = false;
		reqs[i].accepted = false;
		reqs[i].rsped = false;
		reqs[i].rspSeq = std::numeric_limits<int>::max();
	}
	rspSeq = 0;
}

void Room::WaitingObj::reset() {
	std::unique_lock<std::mutex> lock(m);
	resetNoLock();
}

void Room::WaitingObj::req(int index, int fromWho, MeldType indType, int raw) {
	std::unique_lock<std::mutex> lock(m);
	reqs[index].fromWho = fromWho;
	reqs[index].indType = indType;
	reqs[index].raw = raw;
	reqs[index].received = false;
	reqs[index].accepted = false;
	reqs[index].rsped = false;
}

bool Room::WaitingObj::receive(int index) {
	std::unique_lock<std::mutex> lock(m);
	reqs[index].received = true;
	reqs[index].rspSeq = rspSeq;
	rspSeq ++;

	return allRspRcvedNoLock();
}

void Room::WaitingObj::accept(int index, MeldType rspType, std::vector<int> raws) {
	std::unique_lock<std::mutex> lock(m);
	reqs[index].accepted = true;
	reqs[index].indType = rspType;

	if (raws.size() < 2) {
		int tile = reqs[index].raw / 4;
		if (reqs[index].fromWho == index) { //kan from self
			for (int i = 0; i < 4; i ++) {
				raws.push_back(tile * 4 + i);
			}
		} else {
			for (int i = 0; i < 4; i ++) {
				int raw = tile * 4 + i;
				if (raw != reqs[index].raw) {
					raws.push_back(raw);
				}
			}
		}
	}

	reqs[index].raws = std::move(raws);
//	reqs[index].raws = raws;
}


bool Room::WaitingObj::received(int index) {
	std::unique_lock<std::mutex> lock(m);
	return reqs[index].received;
}

bool Room::WaitingObj::accepted(int index){
	std::unique_lock<std::mutex> lock(m);
	return reqs[index].accepted;
}

bool Room::WaitingObj::processed(int index) {
	std::unique_lock<std::mutex> lock(m);
	return reqs[index].rsped;
}

bool Room::WaitingObj::reqed(int index) {
	std::unique_lock<std::mutex> lock(m);
	return (reqs[index].indType != MeldType::Invalid);
}

bool Room::WaitingObj::reqed() {
	std::unique_lock<std::mutex> lock(m);
	for (int i = 0; i < PU::PlayerNum; i ++) {
		if (reqs[i].indType != MeldType::Invalid) {
			return true;
		}
	}

	return false;
}

bool Room::WaitingObj::isAgari(int index) {
//	const static std::set<MeldType> ronInds {
//		MeldType::RonRspInd6, MeldType::RonRspInd7, MeldType::RonRspInd9};

	std::unique_lock<std::mutex> lock(m);
	if (reqs[index].accepted) {
//		return (ronInds.find(reqs[index].indType) != ronInds.end());
		return MeldTypeHelper::IsRonInd(reqs[index].indType);
	}

	return false;
}

bool Room::WaitingObj::isReach(int index) {
	std::unique_lock<std::mutex> lock(m);
	return MeldTypeHelper::IsReachInd(reqs[index].indType);
//	return (reqs[index].indType == 32);
}

int Room::WaitingObj::getRspIndex() {
	int index = -1;
	for (int i = 0; i < PU::PlayerNum; i ++) {
		if (reqs[i].accepted) {
//			if (index < 0) {
//				index = i;
//			} else if (reqs[index].indType > reqs[i].indType){ //chow prior to pong
//				index = i;
//			} else if (reqs[index].indType == reqs[i].indType) {
//				if (reqs[index].rspSeq > reqs[i].rspSeq) {
//					index = i;
//				}
//			}

			if (index < 0) {
				index = i;
			} else if (MeldTypeHelper::Prior2(reqs[index].indType, reqs[i].indType)) {
				index = i;
			} else if (MeldTypeHelper::Prior2(reqs[i].indType, reqs[index].indType)) {
				//nothing
			} else {
				if (reqs[index].rspSeq > reqs[i].rspSeq) {
					index = i;
				}
			}
		}
	}

	return index;
}

int Room::WaitingObj::getNextDistIndex() {
	for (int i = 0; i < PU::PlayerNum; i ++) {
		if (reqed(i)) {
			return (reqs[i].fromWho + 1) % PU::PlayerNum;
		}
	}

	return 0; //impossible
}

//fromWho, indType, raw
std::tuple<int, MeldType, int> Room::WaitingObj::getActInfo(int index) {
	std::unique_lock<std::mutex> lock(m);
	return {reqs[index].fromWho, reqs[index].indType, reqs[index].raw};
}

bool Room::WaitingObj::allRspRcvedNoLock() {
	for (int i = 0; i < PU::PlayerNum; i ++) {
		if (MeldTypeHelper::IsValid(reqs[i].indType) && (!reqs[i].received)) {
			return false;
		}
	}

	return true;
}

bool Room::WaitingObj::allRspRcved() {
	std::unique_lock<std::mutex> lock(m);

	return allRspRcvedNoLock();
}

void Room::WaitingObj::printReqInfo(int index) {
	logger->debug("{} from {}: raw = {}, indType = {}, {}, {}, {}",
			index, reqs[index].fromWho, reqs[index].raw, reqs[index].indType,
			reqs[index].received, reqs[index].accepted, reqs[index].rsped);
}

void Room::WaitingObj::printReqsInfo() {
	for (int i = 0; i < PU::PlayerNum; i ++) {
		printReqInfo(i);
	}
}


Room::Room(uint32_t iSeq)
	: seq(iSeq),
	  roomState(AuthState)
{
	srand(time(NULL));

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	gen = std::mt19937(rd());

	for (int i = 0; i < PU::PlayerNum; i ++) {
		states.push_back(std::move(PlayerState(i, seq)));
	}

//	logger->info("Room {} created", seq);
}

std::shared_ptr<Room> Room::Create(uint32_t iSeq) {
	return std::shared_ptr<Room>(new Room(iSeq));
}

void Room::addClient(int index, std::shared_ptr<ClientConn> client) {
	clients[index] = client;
}

int Room::decideNextOya() {
	return rand() % PU::PlayerNum;
}

int Room::getAbsIndex(const int index, const int clientIndex) const {
	return (index + clientIndex) % PU::PlayerNum;
}
int Room::getRelaIndex(const int index, const int clientIndex) const {
	return (index - clientIndex + PU::PlayerNum) % PU::PlayerNum;
}

void Room::sceneInit() {
	//Seemed nothing
	logger->info("Room {} scene inited", seq);
}

void Room::sendInitMsg(int clientIndex) {
	//<INIT seed="0,0,0,2,2,71" ten="250,250,250,250" oya="2" hai="10,112,68,6,90,8,47,15,89,40,131,69,122"/>
	std::stringbuf buf;
	std::ostream output(&buf);

	output << "<INIT seed=\"0,0,0,2,2,71\" ";
	output << "ten=\"" << std::to_string(tens[getAbsIndex(0, clientIndex)])
				<< "," << std::to_string(tens[getAbsIndex(1, clientIndex)])
				<< "," << std::to_string(tens[getAbsIndex(2, clientIndex)])
				<< "," << std::to_string(tens[getAbsIndex(3, clientIndex)])
				<< "\" ";
	output << "oya=\"" + std::to_string(getAbsIndex(oyaIndex, clientIndex)) + "\" ";
	output << "hai=\"" ;
	for (int i = 0; i < states[clientIndex].acceptRaws.size(); i ++) {
		if (i == 0) {
			output << states[clientIndex].acceptRaws[i];
		} else {
			output << "," << states[clientIndex].acceptRaws[i];
		}
	}

	output << "\" ";
	output << "/>";

	clients[clientIndex]->send(buf.str());
}

void Room::gameInit() {
	/*
	 * 1. create wall
	 * 2. decide oya
	 * 3. No dora
	 * 4. clear board
	 */
	wo.reset();

	reachRaws.clear();
	for (int i = 0; i < PU::PlayerNum; i ++) {
		reachRaws[i] = -1;
	}


	for (int i = 0; i < PU::TotalTileNum; i ++) {
		wall[i] = i;
	}

	std::shuffle(wall.begin(), wall.end(), gen);
//	rndTensor = torch::randperm(136, torch::TensorOptions().dtype(at::kLong));
//	wallData = rndTensor.data_ptr<long>();
//	wallData = rndTensor.to(torch::kLong).data_ptr<long>();
	std::cout << "wall data " << std::endl;
	for (int i = 0; i < 10; i ++) {
		std::cout << wall[i] << ", ";
	}
	std::cout << std::endl;

	oyaIndex = decideNextOya();

	for (int i = 0; i < PU::PlayerNum; i ++) {
		states[i].reset();
	}

	tileIndex = 0;
	for (int i = 0; i < PU::NormTile; i ++) {
		for (int j = 0; j < PU::PlayerNum; j ++) {
//			int raw = wallData[tileIndex]; //random
			int raw = wall[tileIndex];
			int tile = raw / 4;
			tileIndex ++;
			states[j].closeTiles[tile] ++;
			states[j].totalTiles[tile] ++;
			states[j].acceptRaws.push_back(raw);
		}
	}

	logger->info("Room{} game inited", seq);
}

void Room::gameEnd() {
	allReady = 0;
}

bool Room::processOrphanCase() {
	bool isOrphan = false;
	for (int i = 0; i < PU::PlayerNum; i ++) {
		if (PU::IsOrphan(states[i].totalTiles)) {
			isOrphan = true;

			std::vector<S> rspMsgs = PU::GenOrphanMsg(i, tens, states[i].totalTiles);
			for (int j = 0; j < PU::PlayerNum; j ++) {
				clients[j]->send(PU::GenProfMsg());
				clients[j]->send(rspMsgs[j]);
			}
			break;
		}
	}

	return isOrphan;
}

bool Room::processAbortCase() {
	bool isAbort = false;
	for (int i = 0; i < PU::PlayerNum; i ++) {
		if (PU::IsKyushu(states[i].totalTiles)) {
			isAbort = true;

			vector<S> rspMsgs = PU::Gen9YaoMsg(i, states[i].totalTiles, tens);
			for (int j = 0; j < PU::PlayerNum; j ++) {
				clients[j]->send(rspMsgs[j]);
			}

			break;
		}
	}

	return isAbort;
}

namespace {
enum {
	RyuReward = 0,
	AgariReward = 20,
	AgariLoss = 20,
	TsumoReward = 30,
	TsumoLoss = 10,
	ReachPrice = 10,
};


}

void Room::processRyu() {
	logger->info("Room{} processRyu", seq);
	roomState = InitState;
	// <RYUUKYOKU ba="0,0" sc="207,-15,240,15,416,15,137,-15" hai1="19,20,26,41,43,45,49,54,61,62" hai2="44,46,56,63" />
	for (int i = 0; i < PU::PlayerNum; i ++) {
		std::stringbuf buf;
		std::ostream output(&buf);
		output << "<RYUUKYOKU ba=\"0,0\" sc=\"";
		for (int j = 0; j < PU::PlayerNum; j ++) {
			output << tens[getAbsIndex(j, i)] << "," << RyuReward;
			if (j != (PU::PlayerNum - 1)) {
				output << ",";
			}
		}
		output << "\" ";
		output << "hai1=\"\" hai2=\"\" hai3=\"\" hai4=\"\" />";
		clients[i]->send(buf.str());
	}

	gameEnd();
}

void Room::processAgari(int who, int fromWho) {
	logger->info("Room{} processAgari", seq);

	vector<int> deltas(PU::PlayerNum, 0);
	if (who == fromWho) {
		logger->debug("Tsumo: {}", who);
		deltas[who] += TsumoReward;
		for (int i = 0; i < PU::PlayerNum; i ++) {
			if (i != who) {
				deltas[i] -= TsumoLoss;
			}
			if (states[i].reached) {
				logger->debug("Player {} reached", i);
				deltas[who] += ReachPrice;
				deltas[i] -= ReachPrice;
			}
		}
	} else {
		logger->debug("Agari {}, {}", who, fromWho);
		deltas[who] += AgariReward;
		deltas[fromWho] -= AgariLoss;
		for (int i = 0; i < PU::PlayerNum; i ++) {
			if (states[i].reached) {
				logger->debug("Player {} reached", i);
				deltas[who] += ReachPrice;
				deltas[i] -= ReachPrice;
			}
		}
	}

	for (int i = 0; i < PU::PlayerNum; i ++) {
		if (tens[i] + deltas[i] <= 0) {
			roomState = SceneEndState;
			for (int j = 0; j < PU::PlayerNum; j ++) {
				clients[j]->send("<PROF lobby=\"3\" type=\"3\" add=\"0\"/>");
			}
			break;
		}
	}

	if (roomState != SceneEndState) {
		roomState = InitState;
	}

	for (int i = 0; i < PU::PlayerNum; i ++) {
		std::stringbuf buf;
		std::ostream output(&buf);

		output << "<AGARI ba=\"0,2\" hai=\"";
		for (int j = 0; j < states[i].totalTiles.size(); j ++) {
			if (j == 0) {
				output << states[i].totalTiles[j];
			} else {
				output << "," << states[i].totalTiles[j];
			}
		}
		output << "\" ";

		output << "machi=\"\" ten=\"\" yaku=\"\" ";

		output << "who=\"" << getRelaIndex(who, i)
				<< "\" fromWho=\"" << getRelaIndex(fromWho, i) << "\" "
				<< "sc=\"";

		for (int j = 0; j < PU::PlayerNum; j ++) {
			if (j != 0) {
				output << ",";
			}
			int nextClientIndex = getAbsIndex(j, i);
			output << tens[nextClientIndex] << "," << deltas[nextClientIndex];
		}

		output << "\" />";
		clients[i]->send(buf.str());
	}

	for (int i = 0; i < PU::PlayerNum; i ++) {
		tens[i] += deltas[i];
	}

	gameEnd();
}

void Room::distRaw(int clientIndex) {
	if (tileIndex >= (136 - 14)) {
		processRyu();
		return;
	}

	//random
//	int raw = wallData[tileIndex];
	int raw = wall[tileIndex];
	int tile = raw / 4;
	tileIndex ++;
	logger->debug("distRaw {} to {}", raw, clientIndex);

	bool ron = states[clientIndex].checkAgari(raw);

	states[clientIndex].acceptTile(raw);

	//TODO: Check ankan
	if (ron) {
		logger->info("Room{} detected ron {}", seq, clientIndex);
		wo.req(clientIndex, clientIndex, MeldType::Tsumo, raw); //tsumo
		roomState = IndState;
		clients[clientIndex]->send("<T" + std::to_string(raw) + " t=\"16\"/>");
	} else if (states[clientIndex].checkReach()) {
		logger->info("Room{} detected reach {}", seq, clientIndex);
		roomState = ReachState;
		wo.req(clientIndex, clientIndex, MeldType::Reach, raw); //reach
		clients[clientIndex]->send("<T" + std::to_string(raw) + " t=\"32\"/>");
	} else {
		roomState = DropState;
		clients[clientIndex]->send("<T" + std::to_string(raw) + "/>");
	}
	clients[getAbsIndex(1, clientIndex)]->send("<U/>");
	clients[getAbsIndex(2, clientIndex)]->send("<V/>");
	clients[getAbsIndex(3, clientIndex)]->send("<W/>");

	logger->info("Room{}:{} dist {}", seq, clientIndex, raw);
}


void Room::processDropMsg(int clientIndex, S& msg) {
	logger->debug("Room{}:{} processDropMsg {}", seq, clientIndex, msg);
	if (msg.find("<D") == S::npos) {
		logger->error("Room{}:{} DropState unexp: {}", seq, clientIndex, msg);
//		roomState = DropState;
		return;
	}

	vector<S> items = P::ParseItems(msg);
	int raw = P::ParseHead("p=\"", items[1]);
	int tile = raw / 4;
	states[clientIndex].dropTile(raw);


	for (int i = 1; i < PU::PlayerNum; i ++) {
		int nextIndex = getAbsIndex(i, clientIndex);
		if (states[nextIndex].checkAgari(raw)) {
			logger->warn("Room{}:{} detected agari {} by {}", seq, clientIndex, nextIndex, raw);
			wo.req(nextIndex, clientIndex, MeldType::RonByDrop, raw);
		}
	}

	for (int i = 1; i < PU::PlayerNum; i ++) {
		int nextIndex = getAbsIndex(i, clientIndex);
		if (!wo.reqed(nextIndex)) {
			auto meldType = states[nextIndex].checkMeldType(clientIndex, tile);
			logger->debug("Room{}:{} detected meld {} by {}: {}", seq, clientIndex, nextIndex, raw, meldType);
			if ( meldType != MeldType::Invalid) {
				wo.req(nextIndex, clientIndex, meldType, raw);
			}
		}
	}

	logger->debug("Room{}:{} collected waiting reqs:", seq, clientIndex);
	for (int i = 0; i < PU::PlayerNum; i ++) {
		auto [fromWho, meldType, raw] = wo.getActInfo(i);
		logger->debug("who={}, fromWho={}, raw={}, type={}", i, fromWho, raw, meldType);
	}


	bool woReqed = wo.reqed();
	if (woReqed) {
		roomState = IndState;
	} else {
		roomState = DropState;
	}

	for (int i = 0; i < PU::PlayerNum; i ++) {
		int nextClientIndex = getAbsIndex(i, clientIndex);
		int relaIndex = getRelaIndex(clientIndex, nextClientIndex);
//		int relaIndex = (clientIndex - nextClientIndex + PU::PlayerNum) % PU::PlayerNum;
		if (!wo.reqed(nextClientIndex)) {
			clients[nextClientIndex]->send("<" + dropMsgHead[relaIndex] + std::to_string(raw) + "/>");
		} else {
			clients[nextClientIndex]->send("<" + dropMsgHead[relaIndex] + std::to_string(raw) + " t=\""
											+ std::to_string(static_cast<int>(wo.getIndType(nextClientIndex))) + "\"/>");
		}
	}

	if (!woReqed) {
		logger->debug("Room{}:{} {} No special indicator, next round", seq, clientIndex, raw);
		distRaw(getAbsIndex(1, clientIndex));
		return;
	}

}

void Room::processIndMsg(int clientIndex, S& msg) {
	logger->debug("Room{}:{} processIndMsg: {}", seq, clientIndex, msg);
	if (msg.find("<N") == S::npos) {
		logger->error("Room{}:{} IndState unexp: {}", seq, clientIndex, msg);
//		roomState = IndState;
		return;
	}

	if (!wo.reqed(clientIndex)) {
		logger->error("Room{}:{} No corresponding waiting req for this ind {}", seq, clientIndex, msg);
		return;
	}

	vector<S> items = P::ParseItems(msg);
//	bool allRspRcved = wo.receive(clientIndex);
	//Ignore indicator
	if (items.size() <= 1) {
//		logger->debug("items size <= 1, {}", items.size());
		//indicator giveup
	} else {
//		logger->debug("items size = {}", items.size());
		vector<int> raws;
		MeldType indType = MeldType::Invalid;
		for (int i = 0; i < items.size(); i ++) {
//			logger->debug("Parse item {}", items[i]);
			if (items[i].find("type") != S::npos) {
				int type = P::ParseHead("type=\"", items[i]);
				indType = static_cast<MeldType>(type);
			} else if (items[i].find("hai0") != S::npos) {
				int hai = P::ParseHead("hai0=\"", items[i]);
				raws.push_back(hai);
			} else if (items[i].find("hai1") != S::npos) {
				int hai = P::ParseHead("hai1=\"", items[i]);
				raws.push_back(hai);
			} else if (items[i].find("hai2") != S::npos) {
				int hai = P::ParseHead("hai2=\"", items[i]);
				raws.push_back(hai);
			} else if (items[i].find("hai3") != S::npos) {
				int hai = P::ParseHead("hai3=\"", items[i]);
				raws.push_back(hai);
			}
		}
		std::sort(raws.begin(), raws.end());
		for (auto raw: raws) {
			std::cout << raw << ", ";
		}
		wo.accept(clientIndex, indType, std::move(raws));
//		wo.accept(clientIndex, indType, raws);
		auto [fromWho, dummy, raw] = wo.getActInfo(clientIndex);
		logger->info("Room{}:{} received req {} rsp {}", seq, clientIndex, fromWho, raw);
	}

	//put wo.receive in previous position may cause race between receive and race
	//e.g. client1 received, client2 received, client2 gave up, client2 receiveAll
	// client2 getRspIndex returns -1, client1 accepted, but not detected.
	if (!wo.receive(clientIndex)) {
		wo.printReqsInfo();
		return;
	}
	//TODOED: Race here. Two clients may find all received after each rsp processed

	//TODOED: No reach penalty for training network
	bool agariDetected = false;
	for (int i = 0; i < PU::PlayerNum; i ++) {
		if (wo.isAgari(i)) {
			logger->info("Room{}:{} agari detected", seq, clientIndex);

			agariDetected = true;
			processAgari(i, wo.getFromWho(i));
		}
	}
	if (agariDetected) {
		wo.reset();
		return;
	}

	//Process race with some priority
	//3-->chow, 1 --> pong
	int rspWaitIndex = wo.getRspIndex();

	if (rspWaitIndex < 0) {
		logger->debug("Room{}:{} no one wants to meld, to dist", seq, clientIndex);
		int nextClientIndex = wo.getNextDistIndex();
		wo.reset();
		distRaw(nextClientIndex);
	} else {
		auto [fromWho, indType, raw] = wo.getActInfo(rspWaitIndex);
		logger->info("Room{}:{}  {} meld {} from {}", seq, clientIndex, rspWaitIndex, indType, fromWho);

		auto meldRaws = wo.getMeldRaws(rspWaitIndex);
		states[rspWaitIndex].meldRaws(raw, meldRaws);
		int m = PlayerState::GetM(indType, raw, meldRaws);

		wo.reset();
		roomState = DropState;

		for (int j = 0; j < PU::PlayerNum; j ++) {
//			int relaIndex = (rspWaitIndex - j + PU::PlayerNum) % PU::PlayerNum;
			int relaIndex = getRelaIndex(rspWaitIndex, j);
			clients[j]->send("<N who=\"" + std::to_string(relaIndex) + "\" m=\"" + std::to_string(m) + "\" />");
		}

		if (MeldTypeHelper::IsKanRsp(indType)) {
//		if ((actInfo[1] != 1) && (actInfo[1] != 3)) { //kan case
			distRaw(rspWaitIndex);
		}
	}
}

void Room::processReachMsg(int clientIndex, S& msg) {
	//Check reach step2
	if (msg.find("<D") != S::npos) {
		vector<S> items = P::ParseItems(msg);
		int raw = P::ParseHead("p=\"", items[1]);
		int tile = raw / 4;

		if (states[clientIndex].reached && std::get<2>(wo.getActInfo(clientIndex)) == raw) {
			logger->info("Room{}:{}: Received reach response: {}", seq, clientIndex, raw);
			wo.reset();
			//tens adjusted at the end of game
			for (int i = 0; i < PU::PlayerNum; i ++) {
				int nextClientIndex = getRelaIndex(clientIndex, i);
//				int nextClientIndex = (clientIndex - i + PU::PlayerNum) % PU::PlayerNum;
				std::stringbuf buf;
				std::ostream output(&buf);

				output << "<REACH who=\"" << nextClientIndex << "\" ten=\"";
				for (int j = 0; j < PU::PlayerNum; j ++) {
					if (j != 0) {
						output << ",";
					}
					output << tens[getAbsIndex(j, i)];
				}
				output << "\" step=\"2\"/>";

				clients[i]->send(buf.str());
			}

			roomState = DropState; //May not be true
			processDropMsg(clientIndex, msg);
		} else if ((!states[clientIndex].reached) && wo.isReach(clientIndex)) {
			logger->info("Room{}:{}: Decide not to reach", seq, clientIndex);
			wo.reset();
			roomState = DropState;
			processDropMsg(clientIndex, msg);
		} else if (states[clientIndex].reached){
			logger->error("Room{}:{} received un-matched reach statement {} != {}", seq, clientIndex, raw, reachRaws[clientIndex]);
		} else {
			logger->error("Room{}:{} ReachState unexp drop: {}", seq, clientIndex, msg);
		}
	} else if (msg.find("REACH") != S::npos) {
		vector<S> items = P::ParseItems(msg);
		int raw = -1;
		for (int i = 0; i < items.size(); i ++) {
			if (items[i].find("hai") != S::npos) {
				raw = P::ParseHead("hai=\"", items[i]);
				break;
			}
		}

		roomState = ReachState;
		states[clientIndex].reached = true;
		wo.req(clientIndex, clientIndex, MeldType::Reach, raw);
		logger->debug("Room{}:{} declare to reahc by {}", seq, clientIndex, raw);

		for (int i = 0; i < PU::PlayerNum; i ++) {
			int nextClientIndex = getRelaIndex(clientIndex, i);
//			int nextClientIndex = (clientIndex - i + PU::PlayerNum) % PU::PlayerNum;
			clients[i]->send("<REACH who=\"" + std::to_string(nextClientIndex) + "\" step=\"1\"/>");
		}
	}
	else {
		logger->error("Room{}:{} ReachState uexp: {}", seq, clientIndex, msg);
	}
}

void Room::processBye(int clientIndex) {
	logger->debug("Room{}:{} Bye", seq, clientIndex);

	std::unique_lock<std::mutex> lock(bar);
	allBye |= (1 << clientIndex);
	if (allBye == AllReady) {
		this->working = false;
	}
}

bool Room::processAuthMsg(int clientIndex, S& msg) {
	logger->debug("Room{}:{} processAuthMsg: {}", seq, clientIndex, msg);

	//Hard code all responses
	if (msg.find("HELO") != S::npos) {
		S rspMsg = "<HELO uname=\"" + std::to_string(clientIndex) + "\" auth=\"20200425-5c395edd\" />";
		clients[clientIndex]->send(rspMsg);
		return true;
	} else if (msg.find("PXR") != S::npos) {
		S rspMsg = "<LN n=\"BoC1BhS1VK1IS\" j=\"\" g=\"IY3I1OI3o1BU4E8Ms3M1EA12IE4Bk12BQ4I8P1Dq2CN2CB2G1k1S1Y1n2G\"/>";
		clients[clientIndex]->send(rspMsg);
		return true;
	} else if (msg.find("JOIN") != S::npos) {
		S lnRspMsg = "<LN n=\"Bnq1BhK1VF1ID\" j=\"\" g=\"\"/>";
		S goRspMsg = "<GO type=\"1\" lobby=\"0\" gpid=\"\"/>";
		S unRspMsg = "<UN n0=\"\" n1=\"\" n2=\"\" n3=\"\" dan=\"\" rate=\"\" sx=\"\"/>";
		S beginRspMsg = "<TAIKYOKU oya=\"2\" log=\"\"/>";

		clients[clientIndex]->send(lnRspMsg);
		clients[clientIndex]->send(goRspMsg);
		clients[clientIndex]->send(unRspMsg);
		clients[clientIndex]->send(beginRspMsg);
		return true;
	} else if (msg.find("GOK") != S::npos) {
		logger->debug("Room{}:{} GOK", seq, clientIndex);

		std::unique_lock<std::mutex> lock(bar);
		if (allGo == AllReady) {
			//nothing;
		} else {
			allGo |= (1 << clientIndex);
			logger->info("Room{}:{} allGo = {}", seq, clientIndex, allGo);

			if (allGo == AllReady) {
				roomState = InitState;
			}
		}
		return true;
	} else if (msg.find("NEXT") != S::npos) {
		logger->debug("Room{}:{} NEXTREADY", seq, clientIndex);
		std::unique_lock<std::mutex> lock(bar);
		allReady |= (1 << clientIndex);
		return true;
	}
	else {
		logger->debug("Room{}:{}: InitMsg unexpected: {}", seq, clientIndex, msg);
		return false;
	}
}

void Room::processInitMsg(int clientIndex, S& msg) {
	if (msg.find("NEXTREADY") != S::npos) {
		logger->debug("Room{}:{} NEXTREADY", seq, clientIndex);

		bool gameStart = false;
		{
			std::unique_lock<std::mutex> lock(bar);

			if (allReady == AllReady) {
				logger->debug("Room{}:{} All ready. Ignore NEXTREADY msg", seq, clientIndex);
			} else {
				allReady |= (1 << clientIndex);
				if (allReady == AllReady) {
					gameStart = true;
				}
			}
		}

		if (gameStart) {
			logger->debug("Room{}:{} game start", seq, clientIndex);

			gameInit();
			for (int i = 0; i < PU::PlayerNum; i ++) {
				sendInitMsg(i);
			}
			//TODOED: Set nextState in distRaw;
			distRaw(oyaIndex);
		}
	} else {
		logger->error("Room{}:{} InitState unexp msg: {}", seq, clientIndex, msg);
	}
}

void Room::processSceneEndMsg(int clientIndex, S& msg) {
	if (msg.find("BYE") == S::npos) {
		logger->error("Room{}:{} InitState unexp msg: {}", seq, clientIndex, msg);
		return;
	}

	processBye(clientIndex);
}

void Room::processMsg(int clientIndex, S& msg) {
	if (msg.find("<Z") != S::npos) {
//		logger->info("Room{}:{} KA", seq, clientIndex);
		return;
	}

	logger->info("Room{}:{} received msg {}", seq, clientIndex, msg);

	switch(roomState) {
	case AuthState:
		processAuthMsg(clientIndex, msg);
		break;
	case InitState:
		processInitMsg(clientIndex, msg);
		break;
	case DropState:
		processDropMsg(clientIndex, msg);
		break;
	case IndState:
		processIndMsg(clientIndex, msg);
		break;
	case ReachState:
		processReachMsg(clientIndex, msg);
		break;
	case SceneEndState:
		processSceneEndMsg(clientIndex, msg);
		break;
	default:
		processNetErr(clientIndex);
		break;
	}
}

void Room::processNetErr(int clientIndex) {
	std::unique_lock<std::mutex> lock(closeMutex);
	if (working) {
		for(int i = 0; i < PU::PlayerNum; i ++) {
			clients[i]->close();
		}
	}
	roomState = InvalidState;
	working = false;
}

Room::~Room() {
	logger->error("Room {} destructed", seq);
}
