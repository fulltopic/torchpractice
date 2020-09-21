/*
 * tenhoustate.cpp
 *
 *  Created on: Apr 15, 2020
 *      Author: zf
 */


#include "tenhouclient/tenhoustate.h"
#include "tenhouclient/tenhouconsts.h"

#include "tenhouclient/tenhoumsgparser.h"
#include <set>
#include <vector>
#include <map>
#include <torch/torch.h>
#include "../../include/utils/logger.h"

using namespace std;
using namespace torch;

using P = TenhouMsgParser;

TenhouState::~TenhouState() {}

//static const int ReachPos = TenhouConsts::TileNum * 2;
//static const int OyaPos = ReachPos + 1;
//static const int WinPos = OyaPos + 1;
//static const int DummyPos = WinPos + 1;
//static const int StateLen = DummyPos + 1;
//
//static const int ChowAction = TileNum + 1;
//static const int PongAction = ChowAction + 1;
//static const int KaKanAction = PongAction + 1;
//static const int AnKanAction = KaKanAction + 1;
//static const int MinKanAction = AnKanAction + 1;
//static const int ReachAction = MinKanAction + 1;
//static const int RonAction = ReachAction + 1;
//static const int NOOPAction = RonAction + 1;

BaseState::BaseState(int ww, int hh):
		w(ww), h(hh),
		isReach(false),
		isOwner(false),
		logger(Logger::GetLogger()){
	myTiles = vector<bool> (TenhouConsts::TileNum * TenhouConsts::NumPerTile, false);
	innerState = torch::zeros({h, w});
}

BaseState::~BaseState() {}

void BaseState::addTile(int raw) {
	myTiles[raw] = true;
	innerState[0][P::Raw2Tile(raw)] += 1;
//	logger->debug("Add tile: " + raw);
}

//TODO: where to put dropped tile of me?
void BaseState::dropTile(int who, int raw) {
	int tile = P::Raw2Tile(raw);
	if (who == ME) {
		innerState[0][tile] -= 1;
		innerState[1][tile] += 1;
		myTiles[raw] = false;
		logger->debug("Me drop tile {}", raw);
	} else {
		innerState[stateIndex(who)][tile] += 1;
		logger->debug("drop tile {} from {}", raw, who);
	}
}

void BaseState::fixTile(int who, vector<int> raw) {
	if (who == ME) {
		auto data = innerState.accessor<float, 2>();

		if (raw.size() == NumPerTile) {
			for (int i = 0; i < raw.size(); i ++) {
				if (myTiles[raw[i]]) {
					logger->warn("Kan case Removed tile {}", raw[i]);
					data[0][P::Raw2Tile(raw[i])] -= 1;
					myTiles[raw[i]] = false;
				}
			}
			data[0][P::Raw2Tile(raw[0]) + TenhouConsts::TileNum] = 4;
		} else {
			for (int i = 0; i < raw.size(); i ++) {
				if (myTiles[raw[i]]) {
					logger->debug("Removed tile {}", raw[i]);
					data[0][P::Raw2Tile(raw[i])] -= 1;
					myTiles[raw[i]] = false;
					data[0][P::Raw2Tile(raw[i]) + TenhouConsts::TileNum] += 1;
				} else {
					logger->warn("Melt case: {}", raw[i]);
					data[0][P::Raw2Tile(raw[i]) + TenhouConsts::TileNum] += 1;
				}
			}
		}
	} else {
		int playerIndex = stateIndex(who);

		for (int i = 0; i < raw.size(); i ++) {
			innerState[playerIndex][P::Raw2Tile(raw[i]) + TenhouConsts::TileNum] += 1;
		}
	}
}

void BaseState::setReach(int playerIndex) {
	if (playerIndex == ME) {
		isReach = true;
		innerState[0][ReachPos] = 1;
	} else {
		innerState[stateIndex(playerIndex)][ReachPos] = 1;
	}
}

void BaseState::setDora(int dora) {
	doras.insert(dora);
}

void BaseState::reset() {
	myTiles.assign(myTiles.size(), false);
	innerState = innerState.fill_(0);
	doras.clear();

	isReach = false;
	isOwner = false;
}

//vector<int> LstmState::getStealTiles(int type, int raw) {
//	switch (type) {
//	case StealType::ChowType:
//		return getChowTiles(raw);
//	case StealType::PongType:
//		return getPongTiles(raw);
//	case StealType::ReachType:
//		return getReachTiles(raw);
//	}
//
//	return vector<int>();
//}

vector<int> BaseState::getPongTiles(int raw) {
	logger->debug("To get pong tiles for {}", raw);
	int tile = P::Raw2Tile(raw);
	vector<int> tiles;

	for (int i = 0; i < TenhouConsts::NumPerTile; i ++) {
		logger->debug("Try {}", (i + tile * NumPerTile));
		if (myTiles[i + tile * NumPerTile] && ((i + tile * NumPerTile) != raw)) {
			logger->debug("Pushed {}", (i + tile * NumPerTile));
			tiles.push_back(i + tile * NumPerTile);
			if (tiles.size() >= 2) {
				return tiles;
			}
		}
	}

	return tiles;
}

vector<int> BaseState::getChowTiles(int raw) {
	logger->debug("Get chow tiles for {}", raw);
	//TODO: To get use of state to remove realtime counting

	static const vector<int> p1 {
		-2, -1, 1
	};
	static const vector<int> p2 {
		-1, 1, 2
	};

	int tile = P::Raw2Tile(raw);
	int begin = (tile / 9) * 9;
	int end = (begin + 9);
	vector<int> candidates;
	vector<int> firsts;
	for (int i = -2; i <= 0; i ++) {
		if ((tile + i) >= begin && (tile + i + 2) < end) {
			firsts.push_back(tile + i);
		}
	}

	logger->debug("First size: {}", firsts.size());
	if (firsts.size() == 0) {
		return vector<int> ();
	}
	for (int i = 0; i < firsts.size(); i ++) {
		logger->debug("First: {}", firsts[i]);
	}

	map<int, int> tileNums;
	for (int i = tile - 2; i <= tile + 2; i ++) {
		tileNums[i] = 0;
	}
	auto data = innerState.accessor<float, 2>();
	for (int i = firsts[0]; i <= (firsts[firsts.size() - 1] + 2); i ++) {
		tileNums[i] = data[0][i];
	}
	for (auto ite = tileNums.begin(); ite != tileNums.end(); ite ++) {
		logger->debug("Get tile num: {} --> {}", ite->first, ite->second);
	}
//	for (int i = firsts[0]; i < firsts[firsts.size() - 1] + 2; i ++) {
//		int count = 0;
//		for (int j = 0; j < NumPerTile; j ++) {
//			if (myTiles[i * NumPerTile + j]) {
//				count ++;
//			}
//		}
//		tileNums[i] = count;
//	}
//

	for (int i = 0; i < p1.size(); i ++) {
		logger->debug("Try {}, {}, {}", p1[i] + tile, tile, p2[i] + tile);
//		if (tileNums[p1[i] + tile] > tileNums[tile] && tileNums[p2[i] + tile] > tileNums[tile]) {
		if (tileNums[p1[i] + tile] > 0 && tileNums[p2[i] + tile] > 0) {
//			candidates.insert(p1[i] + tile);
//			candidates.insert(p2[i] + tile);
			for (int j = 0; j < NumPerTile; j ++) {
				logger->debug("Try raw: {}", (p1[i] + tile) * NumPerTile + j);
				if (myTiles[(p1[i] + tile) * NumPerTile + j]) {
					candidates.push_back((p1[i] + tile) * NumPerTile + j);
					logger->debug("Pushed {}",(p1[i] + tile) * NumPerTile + j);
					break;
				}
			}
			for (int j = 0; j < NumPerTile; j ++) {
				logger->debug("Try raw: {}", (p2[i] + tile) * NumPerTile + j);
				if (myTiles[(p2[i] + tile) * NumPerTile + j]) {
					candidates.push_back((p2[i] + tile) * NumPerTile + j);
					logger->debug("Pushed {}",(p2[i] + tile) * NumPerTile + j);
					break;
				}
			}
		}
	}

	return candidates;

//	vector<int> rawTiles;
//	sort(candidates.begin(), candidates.end());
//	for (auto ite = candidates.begin(); ite != candidates.end(); ite ++) {
//		for (int i = 0; i < NumPerTile; i ++) {
//			if (myTiles[(*ite) * NumPerTile + i]) {
//				rawTiles.push_back(*ite + i);
//				break;
//			}
//		}
//	}

//	return rawTiles;

}

vector<int> BaseState::getDropCandidates() {
	vector<int> tiles;

	auto stateData = innerState.accessor<float, 2>();
	for (int i = 0; i < TileNum; i ++) {
		if (stateData[0][i] > 0) {
			tiles.push_back(i);
		}
	}
	return tiles;
}

//TODO int --> StealType
vector<int> BaseState::getCandidates(int type, int raw) {
	auto stateData = innerState.accessor<float, 2>();
	logger->debug("Get candidates for {}, {}", type, raw);

	if (type == StealType::DropType) {
		if (isReach) {
			return vector<int> {raw};
		}
		auto tiles = getDropCandidates();
		if ((raw < 0) || (raw >= TileNum)) {
			return tiles;
		}

		if (canKan(P::Raw2Tile(raw), tiles)) {
			Logger::GetLogger()->warn("Could kan tile {}", raw);
		}
		return tiles;
	} else if (type == StealType::ReachType) {
		vector<int> tiles;
		for (int i = 0; i < TileNum; i ++) {
			if ((stateData[0][i] > 0) && (i != P::Raw2Tile(raw))) {
				tiles.push_back(i);
			}
		}
		tiles.push_back(ReachAction);
//		tiles.push_back(NOOPAction); //TODO: Seemed dropping directly means noop
		return tiles;
	} else if (type == StealType::PongType) {
		auto tiles = getDropCandidates();
		tiles.push_back(PongAction);
		tiles.push_back(NOOPAction);
		return tiles;
	} else if (type == StealType::ChowType) {
		auto tiles = getDropCandidates();
		tiles.push_back(ChowAction);
		tiles.push_back(NOOPAction);
		return tiles;
	} else if (type == StealType::PonChowType) {
		auto tiles = getDropCandidates();
		tiles.push_back(PongAction);
		tiles.push_back(ChowAction);
		tiles.push_back(NOOPAction);
	} else if (type == StealType::PonKanType) {
		auto tiles = getDropCandidates();
		tiles.push_back(PongAction);
		tiles.push_back(KaKanAction);
		tiles.push_back(NOOPAction);
	}

	return vector<int>();
}

bool BaseState::is7Pair(const vector<int>& nums) {
	int count = 0;
	for (int i = 0; i < nums.size(); i ++) {
		switch (nums[i]) {
		case 2:
		case 3:
			count ++;
			break;
		case 4:
			count += 2;
			break;
		default:
			break;
			//nothing
		}
	}

	return (count == 6);
}

set<int> BaseState::get7PairTiles(const vector<int>& nums) {
	logger->debug("Get 7 pairs reach tiles");
	set<int> tiles;
	for (int i = 0; i < nums.size(); i ++) {
		if ((nums[i] % 2) > 0) {
			tiles.insert(i);
		}
	}

	return tiles;
}

static set<int> OrphanTiles {
	0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33
};
bool BaseState::is13Orphan(const vector<int>& nums) {


	int count = 0;
	for (int i = 0; i < nums.size(); i ++) {
		if ((OrphanTiles.find(i) != OrphanTiles.end())
				&& (nums[i] > 0)) {
			count ++;
		}
	}
	return (count == 12);
}

set<int> BaseState::get13OrphanTiles(const vector<int>& nums) {
	logger->debug("Get 13 reach tiles ");
	int tile = -1;

	for (int i = 0; i < nums.size(); i ++) {
		if ((nums[i] > 0) && (OrphanTiles.find(i) == OrphanTiles.end())) {
			tile = i;
			break;
		}
	}

	return { tile };
}

static bool mayChow(int tile1, int tile2) {
	return ((tile1 / NumPerCategory) == (tile2 / NumPerCategory))
			&& (-2 <= (tile1 - tile2)) && ((tile1 - tile2) <= 2)
			&& (tile1 < 27) && (tile2 < 27);
}

static bool isChowNb(int tile1, int tile2) {
	return ((tile1 / NumPerCategory) == (tile2 / NumPerCategory))
			&& (-1 <= (tile1 - tile2) <= 1)
			&& (tile1 < 27) && (tile2 < 27);
}

void BaseState::get4GroupTiles(vector<int>& nums, set<int>& tiles, bool hasPair) {
	cout << "Try 4 group " << endl;
	for (int i = 0; i < nums.size(); i ++) {
		cout << nums[i] << ", ";
		if ((i + 1) % 9 == 0) {
			cout << endl;
		}
	}
	cout << endl;

	int count = 0;
	int startIndex = -1;
	int remainTile = 0;
	for (int i = 0; i < nums.size(); i ++) {
		count += nums[i];
		if (nums[i] > 0) {
			if (startIndex < 0) {
				startIndex = i;
			}
			remainTile ++;
		}
	}

	if (count < 2) {
		cout << "Trial failure" << endl;
		return;
	} else if (count == 2) {
		for (int i = startIndex; i < nums.size(); i ++) {
			if (nums[i] > 0) {
				cout << "PUSHED0 " << i << endl;
				tiles.insert(i);
			}
		}
		return;
	} else if (count == 3) {
		switch (remainTile) {
		case 2:
		{
//			if (nums[startIndex] == 1) {
//				cout << "PUSHED1 " << startIndex << endl;
//				tiles.insert(startIndex);
//			} else {
//				for (int i = startIndex; i < nums.size(); i ++){
//					if (nums[i] == 1) {
//						cout << "PUSHED2 " << i << endl;
//						tiles.insert(i);
//						break;
//					}
//				}
//			}
			for (int i = startIndex; i < nums.size(); i ++) {
				if (nums[i] == 1) {
					tiles.insert(i);
				}
			}

			//120 or 210
			if (isChowNb(startIndex, startIndex + 1) && (nums[startIndex] > 0) && (nums[startIndex + 1] > 0)) {
				if (nums[startIndex] > 1) {
					tiles.insert(startIndex);
				} else {
					tiles.insert(startIndex + 1);
				}
			}
			if (isChowNb(startIndex, startIndex + 2) && (nums[startIndex] > 0) && (nums[startIndex + 2]) > 0) {
				if (nums[startIndex] > 1) {
					tiles.insert(startIndex);
				} else {
					tiles.insert(startIndex + 2);
				}
			}
		}
			break;
		case 3:
		{
			vector<int> chowCandidates;
			for (int i = startIndex; i < nums.size(); i ++) {
				if (nums[i] > 0) {
					chowCandidates.push_back(i);
				}
			}

			logger->debug("Test chow {}, {}, {}", chowCandidates[0], chowCandidates[1], chowCandidates[2]);
			if (mayChow(chowCandidates[0], chowCandidates[1])) {
				cout << "PUSHED3 " << chowCandidates[2] << endl;
				tiles.insert(chowCandidates[2]);
			}
			if (mayChow(chowCandidates[1], chowCandidates[2])) {
				cout << "PUSHED4 " << chowCandidates[0] << endl;
				tiles.insert(chowCandidates[0]);
			}
		}
			break;
		default:
			break;
			//impossible
		}
	} else {
		if (!hasPair) {
			for (int i = startIndex; i < nums.size(); i ++) {
				if (nums[i] >= 2) {
					nums[i] -= 2;
					get4GroupTiles(nums, tiles, true);
					nums[i] += 2;
//					break;
				}
			}
		}

		//try pong
		for (int i = startIndex; i < nums.size(); i ++) {
			if (nums[i] >= 3) {
				nums[i] -= 3;
				get4GroupTiles(nums, tiles, hasPair);
				nums[i] += 3;
//				break;
			}
		}

		//try chow
		for (int i = startIndex; i < 25; i ++) {
			if (nums[i] > 0 && nums[i + 1] > 0 && nums[i + 2] > 0
					&& isChowNb(i, i + 1) && isChowNb(i + 1, i + 2)) {
				nums[i] --;
				nums[i + 1] --;
				nums[i + 2] --;
				get4GroupTiles(nums, tiles, hasPair);
				nums[i] ++;
				nums[i + 1] ++;
				nums[i + 2] ++;

//				break;
			}
		}
	}
}

vector<int> BaseState::getReachTiles(int raw) {
	logger->debug("To get reach tiles for {}", raw);
	set<int> reachTiles;

	auto stateData = innerState.accessor<float, 2>();
	vector<int> nums(TileNum, 0);
	for (int i = 0; i < TileNum; i ++) {
		nums[i] = stateData[0][i];
	}
	for(int i = 0; i < TileNum; i ++) {
		cout << nums[i] << ", ";
		if ((i + 1) % 9 == 0) {
			cout << endl;
		}
	}
	cout << endl;

	if (is13Orphan(nums)) {
		reachTiles = get13OrphanTiles(nums);
	} else if (is7Pair(nums)) {
		reachTiles = get7PairTiles(nums);
	} else {
		get4GroupTiles(nums, reachTiles, false);
	}

	cout << "Get reach tiles " << endl;
	for (auto ite = reachTiles.begin(); ite != reachTiles.end(); ite ++) {
		cout << *ite << endl;
	}

	vector<int> rcTiles;
	for (auto ite = reachTiles.begin(); ite != reachTiles.end(); ite ++) {
		for (int i = 0; i < NumPerTile; i ++) {
			if (myTiles[i + *ite * NumPerTile]) {
				rcTiles.push_back(i + *ite * NumPerTile);
			}
		}
	}
	cout << "Get tile candidates: " << endl;
	for (int i = 0; i < rcTiles.size(); i ++) {
		cout << rcTiles[i] << ", ";
	}
	cout << endl;

	for (int i = 0; i < rcTiles.size(); i ++) {
		if (P::Raw2Tile(rcTiles[i]) == raw) {
			return {rcTiles[i]};
		}
	}

	return rcTiles;
}

vector<int> BaseState::getTiles(int type, int raw) {
	static auto logger = Logger::GetLogger();

//	if ((type >= 0) && (type < TileNum)) {
//		logger->debug("To get drop tiles for {}", type);
//		int tile = type;
//		vector<int> tiles;
//		for (int i = 0; i < NumPerTile; i ++) {
//			logger->debug("Try {}", (i + tile * NumPerTile));
//			if (myTiles[i + tile * NumPerTile]) {
//				tiles.push_back(i + tile * NumPerTile);
//				logger->debug("Push {}", (i + tile * NumPerTile));
//			}
//		}
//
//		return tiles;
//	}

	switch (type) {
	case StealType::DropType:
	{
		int tile = raw;
		logger->debug("To get drop tiles for {}", tile);
		vector<int> tiles;
		for (int i = 0; i < NumPerTile; i ++) {
			logger->debug("Try {}", (i + tile * NumPerTile));
			if (myTiles[i + tile * NumPerTile]) {
				tiles.push_back(i + tile * NumPerTile);
				logger->debug("Push {}", (i + tile * NumPerTile));
			}
		}

		return tiles;
	}
	case StealType::ChowType:
		return getChowTiles(raw);
	case StealType::PongType:
		return getPongTiles(raw);
	case StealType::ReachType:
		return getReachTiles(raw);
	default:
		return vector<int>();
	}
}

int BaseState::getChow() { return ChowAction; }
int BaseState::getPong() { return PongAction; }
int BaseState::getReach() { return ReachAction; }

Tensor BaseState::getState(int indType) {
//	int tile = P::Raw2Tile(raw);
	Tensor cpy = innerState.detach().clone(); //TODO: Maybe detach is not necessary

//	switch (indType) {
//	case StealType::ChowType:
//		cpy[1][ChowPos] = 1;
////		if ((tile >= 0) && (tile < TileNum)) {
////			cpy[0][tile] += 1;
////		}
//		break;
//	case StealType::PongType:
//		cpy[1][PongPos] = 1;
////		if ((tile >= 0) && (tile < TileNum)) {
////			cpy[0][tile] += 1;
////		}
//		break;
//	case StealType::ReachType:
//		cpy[1][ReachPos] = 1;
//		break;
//	}

	if ((indType != StealType::DropType) && (indType != StealType::UnknownType)) {
		auto dataPtr = cpy.accessor<float, 2>();
		for (int i = 34; i < 42; i ++) {
			dataPtr[1][i] = 1;
		}
	}

//	return vector<Tensor> ();
	return {cpy.view({1, 1, w * h}).div(4)};
}

//void BaseState::updateState(vector<Tensor> newStates) {
//	//nothing
//}

vector<Tensor> BaseState::endGame() {
	return vector<Tensor> ();
}

bool BaseState::isReached(int playerIndex) {
	auto data = innerState.accessor<float, 2>();
	if (playerIndex == ME) {
		return (data[0][ReachPos] > 0);
	} else {
		return (data[playerIndex + 1][ReachPos] > 0);
	}
}

bool BaseState::canKan(int tile, vector<int>& candidates) {
	if ((tile >= 0) && (tile < 34)) {
		auto data = innerState.accessor<float, 2>();
		logger->debug("cankan for {}: {}, {}", tile, data[0][tile], data[0][tile + TileNum]);
		if ((int)data[0][tile] == NumPerTile) {
			candidates.push_back(AnKanAction);
			candidates.push_back(MinKanAction);
			return true;
		}

		if ((int)data[0][tile + TileNum] == (NumPerTile - 1)) {
			candidates.push_back(KaKanAction);
			return true;
		}
	}

	return false;
}
