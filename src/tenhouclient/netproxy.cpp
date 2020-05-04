/*
 * netproxy.cpp
 *
 *  Created on: Apr 14, 2020
 *      Author: zf
 */

#include "tenhouclient/netproxy.h"
#include "tenhouclient/tenhoumsgparser.h"
#include "tenhouclient/tenhoumsggenerator.h"

#include "tenhouclient/tenhouconsts.h"
#include "tenhouclient/tenhoufsm.h"

#include "nets/a3cnet.h"

#include <vector>
#include <string>

#include <torch/torch.h>

using namespace std;
using namespace torch;

using P = TenhouMsgParser;
using G = TenhouMsgGenerator;
NetProxy::NetProxy(TenhouState& state, TenhouPolicy& iPolicy):
	innerState(state),
	policy(iPolicy),
	logger(Logger::GetLogger()){

}

//TODO: FSM reset?
void NetProxy::reset() {
	innerState.reset();
}

//TODO: Reset
string NetProxy::processInitMsg(string msg) {
	reset();
	policy.reset();

	auto rc = P::ParseInit(msg);
	innerState.setOwner(rc.oyaIndex);

	logger->debug("After parse init ");
	for (int i = 0; i < rc.tiles.size(); i ++) {
		innerState.addTile(rc.tiles[i]);
	}
	logger->debug("After innerState update");

	return StateReturnType::Nothing;
}

string NetProxy::processDropMsg(string msg) {
	auto rc = P::ParseDrop(msg);
	innerState.dropTile(rc.playerIndex, rc.tile);

	return StateReturnType::Nothing;
}

string NetProxy::processDoraMsg(string msg) {
	innerState.setDora(P::ParseDora(msg));

	return StateReturnType::Nothing;
}

string NetProxy::processAccept(string msg) {
	logger->debug("processAccept");
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
	auto output = net.forward(innerState.getState(StealType::DropType));
	int action = policy.getAction(output, candidates);
	logger->debug("Extract action from policy {}", action);
	//kan
	//4 --> ankan
	//5 --> minkan
	if (action >= TileNum) {
		int replyType = 0;
		if (innerState.isMinKanAction(action)) {
			replyType = 4;
		} else if (innerState.isAnKanAction(action)) {
			replyType = 5;
		} else if (innerState.isKaKanAction(action)) {
			replyType = 5;
		}
		string reply = G::AddWrap("N type=\"" + to_string(replyType) + "\" hai=\"" + to_string(rc) + "\" ");
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

string NetProxy::processNMsg(string msg) {
	StealResult rc = P::ParseSteal(msg);

	if (rc.flag >= 0) {
		innerState.fixTile(rc.playerIndex, rc.tiles);
	}

	if (rc.playerIndex == ME) {
		if ((rc.flag != ChowFlag) && (rc.flag != PongFlag)) {
			return StateReturnType::Nothing;
		}
		auto input = innerState.getState(StealType::DropType);
		auto output = net.forward(input);
		auto candidates = innerState.getCandidates(StealType::DropType, 0);
		int action = policy.getAction(output, candidates);
		auto dropTiles = innerState.getTiles(StealType::DropType, action);

		return G::GenDropMsg(dropTiles[0]);
	} else {
		return StateReturnType::Nothing;
	}
}

string NetProxy::processReachMsg(string msg) {
	ReachResult rc = P::ParseReach(msg);
	innerState.setReach(rc.playerIndex);

	return StateReturnType::Nothing;
}

//TODO: Make TenhouState store states for further training
string NetProxy::processChowInd(int fromWho, int raw) {
	innerState.dropTile(fromWho, raw);

	auto input = innerState.getState(StealType::ChowType);
	auto output = net.forward(input);
	auto action = policy.getAction(output, innerState.getCandidates(StealType::ChowType, raw));
	logger->debug("Get action for chow indicator {}", action);

	if (action == ChowAction) {
		vector<int> chowCandidates = innerState.getTiles(StealType::ChowType, raw);
		vector<int> chowTiles = policy.getTiles4Action(output, ChowAction, chowCandidates);
		return G::GenChowMsg(chowTiles);
	} else {
		return G::GenNoopMsg();
	}
}

string NetProxy::processPongInd(int fromWho, int raw) {
	logger->debug("Process pong indicator {}, {}", fromWho, raw);
	//Remove when received N message
	innerState.dropTile(fromWho, raw);

	auto input = innerState.getState(StealType::PongType);
	auto output = net.forward(input);
	auto action = policy.getAction(output, innerState.getCandidates(StealType::PongType, raw));

	//TODO: replace == by function of state object
	if (action == PongAction) {
		vector<int> pongTiles = innerState.getTiles(StealType::PongType, raw);
		return G::GenPongMsg(pongTiles);
	} else {
		return G::GenNoopMsg();
	}
}

string NetProxy::processRonInd(int fromWho, int raw, int type, bool isTsumogiri) {
	//Add action executed in gameend msg
	string reply = G::GenRonMsg(type);
	logger->debug("processRonInd from {} with type {}", fromWho, type);

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

string NetProxy::processGameEndInd(string msg) {
	int reward = 0;
	if (msg.find("AGARI") != string::npos) {
		logger->debug("Process agari message");
		auto rc = P::ParseAgari(msg);
		if (rc.winnerIndex == ME) {
			logger->debug("Process me win ");
			//TODO: generate state copy for further training
			innerState.addTile(rc.machi);
		}
		reward = rc.reward;
	} else if (msg.find("RYU") != string::npos) {
		reward = P::ParseRyu(msg);
	}

	vector<Tensor> inputs = innerState.endGame();
	//TODO: To inject inputs into net back storage
//	policy.reset();

	return StateReturnType::Nothing;
}

string NetProxy::processReachInd(int raw) {
	//TODO: The innerState create copy of state with special flag
	//TODO: without impact on original board state
	//Net to decide if reach
	innerState.addTile(raw);
	auto inputs = innerState.getState(StealType::ReachType);
	auto output = net.forward(inputs);

	auto tiles = innerState.getCandidates(StealType::ReachType, raw);

	//TODO: Pay attention to at::Tensor and torch::Tensor
	auto action = policy.getAction(output, tiles);

	if (innerState.toReach(action)) {
		logger->debug("To reach ");
		innerState.setReach(ME);

//		vector<int> excludes {innerState.getReach(), P::Raw2Tile(raw)};
		//Maybe not needed
		int dropTile = policy.getAction(output, tiles); //Reach has lots of constraints, not to policied
		logger->debug("Get dropTile from policy for reach: {}", dropTile);
		auto dropTiles = innerState.getTiles(StealType::ReachType, dropTile);
		return G::GenReachMsg(dropTiles[0])
				+ StateReturnType::SplitToken
				+ G::GenDropMsg(dropTiles[0]);
	} else {
		logger->debug("Decide not to reach");
//		innerState.addTile(raw); //Added
		//TODO: Check if state deals with drop
		auto dropTiles = innerState.getTiles(StealType::DropType, action);
		return G::GenDropMsg(dropTiles[0]);
	}
}

string NetProxy::processKanInd(int fromWho, int raw) {
	innerState.dropTile(fromWho, raw);

	return G::GenNoopMsg();
}

string NetProxy::processPongKanInd(int fromWho, int raw) {
//	return processPongInd(fromWho, raw);
	innerState.dropTile(fromWho, raw);

	auto input = innerState.getState(StealType::PonKanType);
	auto output = net.forward(input);
	auto action = policy.getAction(output, innerState.getCandidates(StealType::PonKanType, raw)); //TODO: What's candidate?
	logger->debug("Get action for pongkan indicator {}", action);

	if (action == PongAction) {
		vector<int> pongTiles = innerState.getTiles(StealType::PongType, raw);
		return G::GenPongMsg(pongTiles);
	} else if (action == KaKanAction || action == MinKanAction || action == AnKanAction) {
		//TODO: Distinguish them
		return G::GenKanMsg(raw);
	} else {
		return G::GenNoopMsg();
	}
}

string NetProxy::processPongChowInd(int fromWho, int raw) {
	innerState.dropTile(fromWho, raw);

	auto input = innerState.getState(StealType::ChowType);
	auto output = net.forward(input);
	auto action = policy.getAction(output, innerState.getCandidates(StealType::PonChowType, raw));
	logger->debug("Get action for pongchowkan indicator {}", action);

	if (action == ChowAction) {
		vector<int> chowTiles = innerState.getTiles(StealType::ChowType, raw);
		return G::GenChowMsg(chowTiles);
	} else if (action == PongAction) {
		vector<int> pongTiles = innerState.getTiles(StealType::PongType, raw);
		return G::GenPongMsg(pongTiles);
	}	else {
		return G::GenNoopMsg();
	}
}

string NetProxy::processPongChowKanInd(int fromWho, int raw) {
	return processPongChowInd(fromWho, raw);
}

string NetProxy::processIndicatorMsg(string msg) {
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

string NetProxy::processMsg(string msg) {
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
		cout << "Pass message " << endl;
		return "";
	case InvalidMsg:
	default:
		cout << "Unexpected message " << endl;
		return "";
	}
}
