/*
 * tenhoustate.h
 *
 *  Created on: Apr 15, 2020
 *      Author: zf
 */

#ifndef INCLUDE_TENHOUCLIENT_TENHOUSTATE_H_
#define INCLUDE_TENHOUCLIENT_TENHOUSTATE_H_

#include <torch/torch.h>
#include <vector>
#include <set>

#include "../utils/logger.h"
#include "tenhouclient/tenhouconsts.h"

class TenhouState {
public:
	virtual ~TenhouState() = 0;
	virtual torch::Tensor getState(int indType) = 0;
//	virtual void updateState(std::vector<torch::Tensor> newStates) = 0;
	virtual void addTile(int raw) = 0;
	virtual void dropTile(int who, int raw) = 0;
	virtual void fixTile(int who, std::vector<int> raw) = 0;
	virtual std::vector<int> getCandidates(int type, int raw) = 0;
	virtual std::vector<int> getTiles(int type, int raw) = 0;
	virtual void reset() = 0;
	virtual std::vector<torch::Tensor> endGame() = 0;

	virtual void setReach(int playerIndex) = 0;;
	virtual void setOwner(int oya) = 0;
	virtual void setDora(int dora) = 0;
	virtual bool isReached(int playerIndex) = 0;

	virtual bool toChow(int action) = 0;
	virtual bool toPong(int action) = 0;
	virtual bool toReach(int action) = 0;
	virtual int getChow() = 0;
	virtual int getPong() = 0;
	virtual int getReach() = 0;

	virtual bool isKanAction(int action) = 0;
	virtual bool isKaKanAction(int action) = 0;
	virtual bool isAnKanAction(int action) = 0;
	virtual bool isMinKanAction(int action) = 0;

	virtual bool canKan(int tile, std::vector<int>& candidates) = 0;
};

enum LstmStateAction {
	ChowAction = TenhouConsts::TileNum, //34
	PongAction = ChowAction + 1, //35
	KaKanAction = PongAction + 1, //36
	MinKanAction = KaKanAction + 1, //37
	AnKanAction = MinKanAction + 1, //38
	ReachAction = AnKanAction + 1, //39
	RonAction = ReachAction + 1, //40
	NOOPAction = RonAction + 1, //41
};

//TODO: Add a furiten state
enum LstmStatePos {
	ReachPos = TenhouConsts::TileNum * 2,
	OyaPos = ReachPos + 1,
	WinPos = OyaPos + 1,
	DummyPos = WinPos + 1,
	StateLen = DummyPos + 1,

	ChowPos = TileNum,
	PongPos = ChowPos + 1,
	KaKanPos = PongPos + 1,
	AnKanPos = KaKanPos + 1,
	MinKanPos = AnKanPos + 1,
//	ReachPos = MinKanPos + 1,
	RonPos = ReachPos + 1,
	NOOPPos = RonPos + 1,
};

class BaseState: public TenhouState {
protected:

	std::vector<bool> myTiles;
	torch::Tensor innerState;

//	const int seqLen;
	const int w;
	const int h;

	bool isReach;
	bool isOwner;

	std::shared_ptr<spdlog::logger> logger;
	std::set<int> doras;

	inline int stateIndex(const int index) {
		return index + 1;
	}

	std::vector<int> getChowTiles(int raw);
	std::vector<int> getPongTiles(int raw);
	std::vector<int> getReachTiles(int raw);

	std::vector<int> getDropCandidates();

	bool is7Pair(const std::vector<int>& nums);
	std::set<int> get7PairTiles(const std::vector<int>& nums);
	bool is13Orphan(const std::vector<int>& nums);
	std::set<int> get13OrphanTiles(const std::vector<int>& nums);
	void get4GroupTiles(std::vector<int>& nums, std::set<int>& tiles, bool hasPair);

public:
	BaseState(int ww, int hh);
	virtual ~BaseState();

	void setReach(int playerIndex);
	inline void setOwner(int oya) { isOwner = (oya == ME); }
	virtual bool isReached(int playerIndex);
	virtual torch::Tensor getState(int indType);
	//For RNN
//	virtual void updateState(std::vector<torch::Tensor> newStates);

	virtual void addTile(int raw);
	virtual void dropTile(int who, int raw);
	virtual void fixTile(int who, std::vector<int> raw);
	virtual std::vector<int> getTiles(int type, int raw);
	virtual std::vector<int> getCandidates(int type, int raw);
	virtual void reset();
	virtual std::vector<torch::Tensor> endGame();

	inline virtual bool toChow(int action) {
		return (action == ChowAction);
	}
	inline virtual bool toPong(int action) {
		return (action == PongAction);
	}
	inline virtual bool toReach(int action) {
		return (action == ReachAction);
	}
	virtual int getChow();// { return ChowAction; };
	virtual int getPong();// { return PongAction; }
	virtual int getReach();// { return ReachAction; }

	virtual bool canKan(int tile, std::vector<int>& candidates);
	virtual void setDora(int dora);

	inline virtual bool isKanAction(int action) {
		return (action == KaKanAction || action == MinKanAction || action == AnKanAction);
	}
	inline virtual bool isKaKanAction(int action) {
		return action == KaKanAction;
	}
	inline virtual bool isAnKanAction(int action) {
		return action == AnKanAction;
	}
	inline virtual bool isMinKanAction(int action) {
		return action == MinKanAction;
	}
};



#endif /* INCLUDE_TENHOUCLIENT_TENHOUSTATE_H_ */
