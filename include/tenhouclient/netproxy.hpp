/*
 * tenhouproxy.h
 *
 *  Created on: Apr 13, 2020
 *      Author: zf
 */

#ifndef INCLUDE_TENHOUCLIENT_NETPROXY_HPP_
#define INCLUDE_TENHOUCLIENT_NETPROXY_HPP_

#include "tenhouconsts.h"
#include "tenhoustate.h"
#include "randomnet.h"

#include <vector>
#include <string>
//#include <memory>
#include <future>
#include <condition_variable>
#include <mutex>

#include <torch/torch.h>

#include "utils/logger.h"
#include "policy/tenhoupolicy.h"

#include "tenhouclient/tenhoumsgparser.h"
#include "tenhouclient/tenhoumsggenerator.h"

#include "tenhouclient/tenhouconsts.h"
//#include "tenhouclient/tenhoufsm.h"
#include "fsmtypes.h"

template <typename T>
struct HasCreateHState {
	template<typename U, torch::Tensor (U::*)()> struct SFINEA {};
	template<typename U> static char Test(SFINEA<U, &U::createHState>*);
	template<typename U> static int Test(...);
	static const bool Has = (sizeof(Test<T>(0)) == sizeof(char));
};


template <int capacity>
struct StateStoreType {
	using StoreDataType = std::vector<std::vector<torch::Tensor>>;

	uint32_t curSeq;
	int curIndex;

	uint32_t lastReadSeq;

	StoreDataType trainStates;
	StoreDataType trainHStates;
	StoreDataType trainLabels; //action executed
	StoreDataType trainActions; //action calculated
	std::vector<float> rewards;


	StateStoreType(): curSeq(0), curIndex(0), lastReadSeq(0 - 1),
			trainStates(capacity, std::vector<torch::Tensor>()),
			trainHStates(capacity, std::vector<torch::Tensor>()),
			trainLabels(capacity, std::vector<torch::Tensor>()),
			trainActions(capacity, std::vector<torch::Tensor>()),
			rewards(capacity, 0.0f)
	{
	}

	void reset() {
		curSeq ++;
		curIndex = curSeq % capacity;

		trainStates[curIndex].clear();
		trainHStates[curIndex].clear();
		trainLabels[curIndex].clear();
		trainActions[curIndex].clear();
		rewards[curIndex] = 0.0f;
	}

	//TODO: Make it more dependable
	StoreDataType getLastData() {
		if (lastReadSeq == (curSeq - 1)) {
			return {};
		}

		uint32_t lastIndex = (curSeq - 1) % capacity;
		return {trainStates[lastIndex], trainHStates[lastIndex], trainLabels[lastIndex], trainActions[lastIndex], {torch::tensor(rewards[lastIndex])}};
	}

	int getCurIndex () {
		return curIndex;
	}
};

template <typename NetType>
class NetProxy {

enum NetStatus {
	Running = 0,
	Updating = 1,
	Detected = 2,
};

private:
	const std::string name;
	NetStatus netStatus;
	std::mutex netStatusMutex;
	std::condition_variable netStatusCond;
//	torch::Tensor board;
//TODO: Remove reference
	TenhouState& innerState;

	//For training
//	std::vector<torch::Tensor> trainStates;
//	std::vector<torch::Tensor> trainActions;
//	std::vector<torch::Tensor> trainLabels;

//	torch::Tensor rnnHidden;
//	torch::Tensor rnnState;
	bool isGru;
	torch::Tensor gruHState;
	int step;
	int gameSeq;
//	std::vector<torch::Tensor> trainHStates;
	//TODO: To get reward of each match
	StateStoreType<2> storage;

	//RandomNet net;
	TenhouPolicy& policy;
	std::shared_ptr<NetType> net;

	std::shared_ptr<spdlog::logger> logger;
	std::shared_ptr<std::promise<bool>> statusPromise;

	std::string processInitMsg(std::string msg);
	std::string processDoraMsg(std::string msg);
	std::string processDropMsg(std::string msg);
	std::string processAccept(std::string msg);
	std::string processNMsg(std::string msg);
	std::string processReachMsg(std::string msg);
	std::string processIndicatorMsg(std::string msg);
	std::string processReachInd(int raw);
	std::string processChowInd(int fromWho, int raw);
	std::string processPongInd(int fromWho, int raw);
	std::string processKanInd(int fromWho, int raw);
	std::string processPongKanInd(int fromWho, int raw);
	std::string processPongChowInd(int fromWho, int raw);
	std::string processPongChowKanInd(int fromWho, int raw);
	std::string processRonInd(int fromWho, int raw, int type, bool isTsumogiri);
	std::string processGameEndInd(std::string msg);

	torch::Tensor updateNet(int indType);
	void updateLabelStore(int label);
	void updateReward(float reward);

//	template<typename T>
	void initGru(std::true_type) {
		logger->error("initGru");
		isGru = true;
		gruHState = net->createHState();
		step = 0;
	}

//	template<typename T>
	void initGru(std::false_type){
		logger->error("init non gru");
	};

public:
	NetProxy(const std::string name, std::shared_ptr<NetType> net, TenhouState& state, TenhouPolicy& iPolicy);
	NetProxy(const NetProxy& other) = delete;

	~NetProxy() = default;
	std::string processMsg(std::string msg);
	void reset();
	void updateNet(std::shared_ptr<NetType> net);

	bool isUpdating() { return netStatus == Updating; }
	bool isRunning() { return netStatus == Running; }
	bool setDetected();
	void setUpdating(std::shared_ptr<std::promise<bool>> promiseObj);
	void setRunning(std::shared_ptr<NetType> newNet);

	std::vector<std::vector<torch::Tensor>> getStates();

	inline std::string getName() { return name; }
};


using P = TenhouMsgParser;
using G = TenhouMsgGenerator;


template<class NetType>
void NetProxy<NetType>::updateLabelStore(int label) {
	int curIndex = storage.getCurIndex();
	storage.trainLabels[curIndex].push_back(torch::tensor(label));
}

template<class NetType>
void NetProxy<NetType>::updateReward(float reward) {
	storage.rewards[storage.curIndex] = reward;
}

template <class NetType>
NetProxy<NetType>::NetProxy(const std::string netName, std::shared_ptr<NetType> iNet, TenhouState& state, TenhouPolicy& iPolicy):
	name(netName),
	netStatus(Running),
	innerState(state),
	isGru(false),
	step(0),
	gameSeq(0),
	policy(iPolicy),
	net(iNet),
	logger(Logger::GetLogger()){

	initGru(std::integral_constant<bool, HasCreateHState<NetType>::Has>());
}

template <class NetType>
torch::Tensor NetProxy<NetType>::updateNet(int indType) {
	const int curIndex = storage.getCurIndex();

	if (isGru) {
		logger->debug("updateNet gru");
		torch::Tensor state = innerState.getState(indType);

		storage.trainStates[curIndex].push_back(state.detach().clone()); //TODO: Maybe detach not necessary
		storage.trainHStates[curIndex].push_back(gruHState.detach().clone());

		std::vector<torch::Tensor> netOutput = net->forward(std::vector{state, gruHState, torch::tensor(step)});
		// get max index
		torch::Tensor actProb = netOutput[0];
		actProb = actProb.clamp(1.21e-7, 1.0f - 1.21e-7);
		torch::Tensor action = actProb.multinomial(1, false);
		storage.trainActions[curIndex].push_back(action);

		gruHState = netOutput[1];
		step ++;
		return netOutput[0];
	} else {
		logger->debug("updateNet non gru");
		torch::Tensor state = innerState.getState(indType);

		storage.trainStates[curIndex].push_back(state.detach().clone()); //TODO: Maybe detach not necessary

		std::vector<torch::Tensor> netOutput = net->forward(std::vector{state});

		torch::Tensor actProb = netOutput[0];
		actProb = actProb.clamp(1.21e-7, 1.0f - 1.21e-7);
		torch::Tensor action = actProb.multinomial(1, false);
		storage.trainActions[curIndex].push_back(action);

		return netOutput[0];
	}
}



//TODO: FSM reset?
template <class NetType>
void NetProxy<NetType>::reset() {
	innerState.reset();
	policy.reset();
//	net->reset();

	initGru(std::integral_constant<bool, HasCreateHState<NetType>::Has>());

}

template <class NetType>
std::vector<std::vector<torch::Tensor>> NetProxy<NetType>::getStates() {
	return std::move(storage.getLastData());
}

//TODO: Reset
template <class NetType>
std::string NetProxy<NetType>::processInitMsg(std::string msg) {
	gameSeq ++;

	reset();
//	policy.reset();

	auto rc = P::ParseInit(msg);
	innerState.setOwner(rc.oyaIndex);

	logger->debug("After parse init ");
	for (int i = 0; i < rc.tiles.size(); i ++) {
		innerState.addTile(rc.tiles[i]);
	}
	logger->debug("After innerState update");

	return StateReturnType::Nothing;
}

template <class NetType>
std::string NetProxy<NetType>::processDropMsg(std::string msg) {
	auto rc = P::ParseDrop(msg);
	innerState.dropTile(rc.playerIndex, rc.tile);

	return StateReturnType::Nothing;
}

template <class NetType>
std::string NetProxy<NetType>::processDoraMsg(std::string msg) {
	innerState.setDora(P::ParseDora(msg));

	return StateReturnType::Nothing;
}

template <class NetType>
std::string NetProxy<NetType>::processAccept(std::string msg) {
	logger->debug("processAccept {}", msg);
	auto rc = P::ParseAccept(msg);
	logger->debug("Accept rc {}", rc);

	if (innerState.isReached(ME)) {
		return G::GenDropMsg(rc);
	}

	innerState.addTile(rc);
	logger->debug("Added tile for accept");

	auto candidates = innerState.getCandidates(StealType::DropType, rc);
	logger->debug("After candidates {}", candidates.size());
	for (int i = 0; i < candidates.size(); i ++) {
		logger->debug("Get candidate {}", candidates[i]);
	}
	//TODO: state take consideration of kan
//	auto output = net->forward(innerState.getState(StealType::DropType));
	torch::Tensor output = updateNet(StealType::DropType);

	int action = policy.getAction(output, candidates);
	updateLabelStore(action); //TODO: For rl
	logger->debug("Extract action from policy {}", action);
	//kan
	//4 --> ankan
	//5 --> minkan
	if (action >= TileNum) {
		int replyType = 0;
		if (innerState.isMinKanAction(action)) {
			replyType = 4;
		} else if (innerState.isAnKanAction(action)) {
			replyType = 4;
		} else if (innerState.isKaKanAction(action)) {
			replyType = 5;
		}
		std::string reply = G::AddWrap("N type=\"" + std::to_string(replyType) + "\" hai=\"" + std::to_string(rc) + "\" ");
		logger->debug("Get reply for kan action {}", reply);
		return reply;
//		string reply = G::AddWrap(string(""));
//		reply += StateReturnType::SplitToken;

//
//		auto dropAction = policy.getAction(output, vector<int>(candidates[0], candidates[TileNum]));
//		logger->debug("Get drop action {}", dropAction);
//		auto dropTile = innerState.getTiles(StealType::DropType, dropAction);
//		logger->debug("Get kan drop tile {}", dropAction);
//		return reply + G::GenDropMsg(dropTile[0]);
	} else {
		logger->debug("Get action {}", action);
		auto dropTile = innerState.getTiles(StealType::DropType, action);
		logger->debug("Get drop tiles {}", dropTile.size());

		if (action == P::Raw2Tile(rc)) {
			return G::GenDropMsg(rc);
		}

		return G::GenDropMsg(dropTile[0]);
	}
}

template <class NetType>
std::string NetProxy<NetType>::processNMsg(std::string msg) {
	StealResult rc = P::ParseSteal(msg);

	if (rc.flag >= 0) {
		innerState.fixTile(rc.playerIndex, rc.tiles);
	}

	if (rc.playerIndex == ME) {
		//TODO: Other indicators are also important
		if ((rc.flag != ChowFlag) && (rc.flag != PongFlag)) {
			return StateReturnType::Nothing;
		}

		//zf: netforward
//		auto input = innerState.getState(StealType::DropType);
//		auto output = net->forward(input);
		torch::Tensor output = updateNet(StealType::DropType);

		auto candidates = innerState.getCandidates(StealType::DropType, -1);
		int action = policy.getAction(output, candidates);
		auto dropTiles = innerState.getTiles(StealType::DropType, action);
		updateLabelStore(action); //TODO: For rl

		return G::GenDropMsg(dropTiles[0]);
	} else {
		return StateReturnType::Nothing;
	}
}

template <class NetType>
std::string NetProxy<NetType>::processReachMsg(std::string msg) {
	ReachResult rc = P::ParseReach(msg);
	innerState.setReach(rc.playerIndex);

	return StateReturnType::Nothing;
}

//TODO: Make TenhouState store states for further training
template <class NetType>
std::string NetProxy<NetType>::processChowInd(int fromWho, int raw) {
	innerState.dropTile(fromWho, raw);

	//zf: netForward
//	auto input = innerState.getState(StealType::ChowType);
//	auto output = net->forward(input);
	torch::Tensor output = updateNet(StealType::ChowType);

	auto action = policy.getAction(output, innerState.getCandidates(StealType::ChowType, raw));
	updateLabelStore(action); //For rl
	logger->info("Get action for chow indicator {}", action);

	if (action == ChowAction) {
		std::vector<int> chowCandidates = innerState.getTiles(StealType::ChowType, raw);
		std::vector<int> chowTiles = policy.getTiles4Action(output, ChowAction, chowCandidates, raw);
		return G::GenChowMsg(chowTiles);
	} else {
		return G::GenNoopMsg();
	}
}

template <class NetType>
std::string NetProxy<NetType>::processPongInd(int fromWho, int raw) {
	logger->debug("Process pong indicator {}, {}", fromWho, raw);
	//Remove when received N message
	innerState.dropTile(fromWho, raw);

	//zf: netForward
//	auto input = innerState.getState(StealType::PongType);
//	auto output = net->forward(input);
	torch::Tensor output = updateNet(StealType::PongType);

	auto action = policy.getAction(output, innerState.getCandidates(StealType::PongType, raw));
	updateLabelStore(action); //TODO: For rl

	//TODO: replace == by function of state object
	logger->info("Get action for pong indicator {}", action);
	if (action == PongAction) {
		std::vector<int> pongTiles = innerState.getTiles(StealType::PongType, raw);
		return G::GenPongMsg(pongTiles);
	} else {
		return G::GenNoopMsg();
	}
}

template <class NetType>
std::string NetProxy<NetType>::processRonInd(int fromWho, int raw, int type, bool isTsumogiri) {
	//Add action executed in gameend msg
	std::string reply = G::GenRonMsg(type);
	logger->info("processRonInd from {} with type {}", fromWho, type);

	if ((type % 16) == 0) {
		return reply;
	}
////	if (fromWho == 3)
//	if (isTsumogiri) { //Don't know rule
//		reply += StateReturnType::SplitToken + G::GenNoopMsg();
//	} else {
////		type -= 8;
//		if ((type - 8) == StealType::PongType) {
//			vector<int> pongTiles = innerState.getTiles(StealType::PongType, raw);
//			reply += (StateReturnType::SplitToken + G::GenPongMsg(pongTiles));
//		} else if ((type - 8) == StealType::ChowType) {
//			vector<int> chowTiles = innerState.getTiles(StealType::ChowType, raw);
//			reply += (StateReturnType::SplitToken + G::GenChowMsg(chowTiles));
//		}
////		else if (type == 9) {
////			//nothing
////		}
//		else {
//			reply += StateReturnType::SplitToken + G::GenNoopMsg();
//		}
//	}

	reply += StateReturnType::SplitToken + G::GenNoopMsg();
	return reply;
}

//TODO: Failed to extract reward
template <class NetType>
std::string NetProxy<NetType>::processGameEndInd(std::string msg) {
	logger->info("End of game {}", gameSeq);

	int reward = 0;
	if (msg.find("AGARI") != std::string::npos) {
		logger->debug("Process agari message");
		auto rc = P::ParseAgari(msg);
		if (rc.winnerIndex == ME) {
			logger->debug("Process me win ");
			//TODO: generate state copy for further training
			innerState.addTile(rc.machi);
		}
		reward = rc.reward;
	} else if (msg.find("RYU") != std::string::npos) {
		reward = P::ParseRyu(msg);
	}

	logger->info("Reward of the game: {}", reward);

//	std::vector<torch::Tensor> inputs = innerState.endGame();
	updateReward(reward);
	storage.reset();

//	policy.reset();

	return StateReturnType::Nothing;
}

template <class NetType>
std::string NetProxy<NetType>::processReachInd(int raw) {
	//TODO: The innerState create copy of state with special flag
	//TODO: without impact on original board state
	//Net to decide if reach
	innerState.addTile(raw);

	//zf: netForward
//	auto inputs = innerState.getState(StealType::ReachType);
//	auto output = net->forward(inputs);
	torch::Tensor output = updateNet(StealType::ReachType);

	auto tiles = innerState.getCandidates(StealType::ReachType, raw);

	auto action = policy.getAction(output, tiles);
	updateLabelStore(action); //For rl

	if (innerState.toReach(action)) {
		logger->debug("To reach ");
		innerState.setReach(ME);



//		int dropTile = policy.getAction(output, tiles); //Reach has lots of constraints, not to policied
//		logger->debug("Get dropTile from policy for reach: {}", dropTile);
//		auto dropTiles = innerState.getTiles(StealType::ReachType, dropTile);
		//TODO: To optimize
		auto reachRaws = innerState.getTiles(StealType::ReachType, -1);
		std::set<int> reachTiles;
		for (int i = 0; i < reachRaws.size(); i ++) {
			reachTiles.insert(P::Raw2Tile(reachRaws[i]));
		}
		int dropTile = policy.getAction(output, std::vector<int>(reachTiles.begin(), reachTiles.end()));
		auto dropTiles = innerState.getTiles(StealType::DropType, dropTile);

		return G::GenReachMsg(dropTiles[0])
				+ StateReturnType::SplitToken
				+ G::GenDropMsg(dropTiles[0]);
	} else {
		//TODO: The corresponding action = 41
		logger->debug("Decide not to reach");
//		innerState.addTile(raw); //Added
		//TODO: Check if state deals with drop
		auto dropTiles = innerState.getTiles(StealType::DropType, action);
		return G::GenDropMsg(dropTiles[0]);
	}
}

template <class NetType>
std::string NetProxy<NetType>::processKanInd(int fromWho, int raw) {
	innerState.dropTile(fromWho, raw);

	return G::GenNoopMsg();
}

template <class NetType>
std::string NetProxy<NetType>::processPongKanInd(int fromWho, int raw) {
//	return processPongInd(fromWho, raw);
	innerState.dropTile(fromWho, raw);

	//zf: netForward
//	auto input = innerState.getState(StealType::PonKanType);
//	auto output = net->forward(input);
	torch::Tensor output = updateNet(StealType::PonKanType);

	auto action = policy.getAction(output, innerState.getCandidates(StealType::PonKanType, raw)); //TODO: What's candidate?
	updateLabelStore(action); //TODO: For rl
	logger->debug("Get action for pongkan indicator {}", action);

	if (action == PongAction) {
		std::vector<int> pongTiles = innerState.getTiles(StealType::PongType, raw);
		return G::GenPongMsg(pongTiles);
	} else if (action == KaKanAction || action == MinKanAction || action == AnKanAction) {
		//TODO: Distinguish them
		return G::GenKanMsg(raw);
	} else {
		return G::GenNoopMsg();
	}
}

template <class NetType>
std::string NetProxy<NetType>::processPongChowInd(int fromWho, int raw) {
	innerState.dropTile(fromWho, raw);

	//zf: netForward
//	auto input = innerState.getState(StealType::ChowType);
//	auto output = net->forward(input);
	torch::Tensor output = updateNet(StealType::ChowType);

	auto action = policy.getAction(output, innerState.getCandidates(StealType::PonChowType, raw));
	updateLabelStore(action); //TODO: For rl
	logger->debug("Get action for pongchowkan indicator {}", action);

	if (action == ChowAction) {
		std::vector<int> chowTiles = innerState.getTiles(StealType::ChowType, raw);
		return G::GenChowMsg(chowTiles);
	} else if (action == PongAction) {
		std::vector<int> pongTiles = innerState.getTiles(StealType::PongType, raw);
		return G::GenPongMsg(pongTiles);
	}	else {
		return G::GenNoopMsg();
	}
}

template <class NetType>
std::string NetProxy<NetType>::processPongChowKanInd(int fromWho, int raw) {
	return processPongChowInd(fromWho, raw);
}

template <class NetType>
std::string NetProxy<NetType>::processIndicatorMsg(std::string msg) {
	auto rc = P::ParseStealIndicator(msg);
	auto stealType = P::GetIndType(msg);
	logger->debug("process indicator msg for {} as {}", rc.type, stealType);

	switch (stealType) {
	case StealType::ChowType:
		return processChowInd(rc.who, rc.tile);
	case StealType::PongType:
		return processPongInd(rc.who, rc.tile);
	case StealType::ReachType:
		return processReachInd(rc.tile);
	case StealType::RonType:
		return processRonInd(rc.who, rc.tile, rc.type, P::IsTsumogiriMsg(msg));
	case StealType::PonChowKanType:
		return processPongChowKanInd(rc.who, rc.tile);
	case StealType::PonChowType:
		return processPongChowInd(rc.who, rc.tile);
	case StealType::PonKanType:
		return processPongKanInd(rc.who, rc.tile);
	case StealType::KanType:
		return processKanInd(rc.who, rc.tile);
//	case StealType::
	default:
		logger->error("Received unparsed indicator {}", msg);
		return "";
	}
}

template <class NetType>
std::string NetProxy<NetType>::processMsg(std::string msg) {
	GameMsgType msgType = P::GetMsgType(msg);
	logger->debug("Received message type: {}", msgType);
	//TODO: IndMsg
	switch (msgType) {
	case InitMsg:
		logger->debug("To process init msg");
		return processInitMsg(msg);
	case DoraMsg:
		return processDoraMsg(msg);
	case DropMsg:
		return processDropMsg(msg);
	case AcceptMsg:
		return processAccept(msg);
	case IndicatorMsg:
		return processIndicatorMsg(msg);
	case NMsg:
		return processNMsg(msg);
	case ReachMsg:
		return processReachMsg(msg);
	case GameEndMsg:
		return processGameEndInd(msg);
	case SilentMsg:
		logger->debug("Pass message: {}", msg);
		return "";
	case InvalidMsg:
	default:
		logger->error("Unexpected message: {}", msg);
		return "";
	}
}

template <class NetType>
void NetProxy<NetType>::setUpdating(std::shared_ptr<std::promise<bool>> promiseObj) {
	logger->warn("NetProxy to set updating: {}", getName());
	netStatus = Updating;

//	std::unique_lock<std::mutex> lock(netStatusMutex);
//	while(netStatus != Detected) {
//		netStatusCond.wait(lock);
//	}
//
//	promiseObj->set_value(true);
//	logger->warn("Network detected updating: {}", getName());
//	return;
	statusPromise = promiseObj;
	logger->warn("Network set updating ");
	return;
}

template <class NetType>
bool NetProxy<NetType>::setDetected() {
	logger->warn("NetProxy try to set detected: {}", getName());
//	std::unique_lock<std::mutex> lock(netStatusMutex);
//
//	if (netStatus == Updating) {
//		netStatus = Detected;
//		netStatusCond.notify_one();
//		logger->warn("NetProxy notified detected: {}", getName());
//		return true;
//	} else if (netStatus == Detected) {
//		logger->warn("NetProxy had been notified by fsm: {}", getName());
//		return false;
//	}	else {
//		logger->info("NetProxy is running: {}", getName());
//		return false;
//	}

	if (netStatus == Updating) {
		if (!statusPromise) {
			logger->error("NetProxy no promise set: {}", getName());
			return false;
		} else {
			logger->warn("NetProxy to set detected: {}", getName());
			netStatus = Detected;
			statusPromise->set_value(true);
			statusPromise.reset();
			return true;
		}
	} else if (netStatus == Detected) {
		logger->warn("NetProxy had been notified by fsm: {}", getName());
		return false;
	} else {
		logger->info("NetProxy is running: {}", getName());
		return false;
	}
}

template <class NetType>
void NetProxy<NetType>::setRunning(std::shared_ptr<NetType> newNet) {
	logger->warn("NetProxy set running again: {}", getName());
	net = newNet;
	netStatus = Running;
}


#endif /* INCLUDE_TENHOUCLIENT_NETPROXY_HPP_ */
