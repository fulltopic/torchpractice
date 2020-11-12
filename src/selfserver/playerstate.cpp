/*
 * playerstate.cpp
 *
 *  Created on: Oct 31, 2020
 *      Author: zf
 */


#include "selfserver/playerstate.h"
#include "selfserver/agarichecker.h"

#include "utils/logger.h"

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <sstream>

using std::string;
using std::vector;

namespace {
	auto logger = Logger::GetLogger();
}

PlayerState::PlayerState(int index, int rIndex)
	: closeTiles(34, 0),
	  dropTiles(34, 0),
	  totalTiles(34, 0),
	  myIndex(index),
	  roomIndex(rIndex),
	  closed(true),
	  reached(false),
	  agariRaw(-1),
	  fromWho (-1)
{

}

void PlayerState::reset() {
	meldTiles.clear();
	closeTiles.assign(34, 0);
	dropTiles.assign(34, 0);
	totalTiles.assign(34, 0);
	acceptRaws.clear();

	closed = true;
	reached = false;
	agariRaw = -1;
	fromWho = -1;

	combs.clear();
	poses.clear();
}

void PlayerState::acceptTile (int raw) {
	int tile = raw / 4;
	closeTiles[tile] += 1;
	totalTiles[tile] += 1;
	logger->debug("Player {}:{} accepted {}", roomIndex, myIndex, raw);
}

void PlayerState::dropTile (int raw) {
	int tile = raw / 4;
	closeTiles[tile] -= 1;
	totalTiles[tile] -= 1;
	dropTiles[tile] += 1;
	logger->debug("Player {}:{} dropped {}", roomIndex, myIndex, raw);
}

//void PlayerState::meldTile (int raw) {
//	//TODO;
//}

//bool PlayerState::checkAgari(int raw) {
//	int tile = raw / 4;
//	totalTiles[tile] += 1;
//
//	auto keys = AgariChecker::GetKey(totalTiles);
//	int key = std::get<0>(keys);
//	if (!AgariChecker::IsAgari(key)) {
//		return false;
//	}
//
//	//TODO: These moves are OK and effective?
//	poses = std::move(std::get<1>(keys));
//	combs = std::move(AgariChecker::GetCombs(key));
//
//	totalTiles[tile] -= 1;
//
//	return true;
//}

namespace {
std::vector<std::vector<std::vector<int>>> kanCombs{
	{{0}},
	{{0}, {1}, {0, 1}},
	{{0}, {1}, {2}, {0, 1}, {0, 2}, {1, 2}, {0, 1, 2}},
	{{0}, {1}, {2}, {3}, {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3},
			{0, 1, 2}, {0, 2, 3}, {1, 2, 3}, {0, 1, 2, 3}}
};

void printVec(std::string comment, const std::vector<int>& data) {
	std::stringbuf buf;
	std::ostream output(&buf);

	for (int i = 0; i < data.size(); i ++) {
		output << data[i] << ",";
	}
	output << std::endl;
	logger->debug("{}: \n {}", comment, buf.str());
}
}

bool PlayerState::checkAgari(int raw) {
	logger->debug("Player {}:{} check agari {}", roomIndex, myIndex, raw);
	int tile = raw / 4;
	totalTiles[tile] += 1;
	printVec("totalTiles ", totalTiles);

	auto keyAndPos = AgariChecker::GetInst()->getKey(totalTiles);
	int key = std::get<0>(keyAndPos);
	if (AgariChecker::GetInst()->isAgari(key)) {
		totalTiles[tile] --;
		return true;
	}

	if (Is7Pair(totalTiles)) {
		totalTiles[tile] --;
		return true;
	}

	std::vector<int> kanTiles;
	for (int i = 0; i < totalTiles.size(); i ++) {
		if (totalTiles[i] == 4) {
			kanTiles.push_back(i);
		}
	}
	int kanNum = kanTiles.size() - 1;
	if (kanNum < 0) {
		totalTiles[tile] --;
		return false;
	}


	for (int i = 0; i < kanCombs[kanNum].size(); i ++) {
		for (int j = 0; j < kanCombs[kanNum][i].size(); j ++) {
			totalTiles[kanTiles[kanCombs[kanNum][i][j]]] --;
		}
		printVec("check kan agari", totalTiles);
		auto keys = AgariChecker::GetInst()->getKey(totalTiles);
		int key = std::get<0>(keys);
//		int key = std::get<0>(AgariChecker::GetInst()->getKey(totalTiles));
		bool isAgari = AgariChecker::GetInst()->isAgari(key);

		for (int j = 0; j < kanCombs[kanNum][i].size(); j ++) {
			totalTiles[kanTiles[kanCombs[kanNum][i][j]]] ++;
		}
		if (isAgari) {
			totalTiles[tile] --;
			return true;
		}
	}

//	for (int i = 0; i < totalTiles.size(); i ++) {
//		if (totalTiles[i] == 4) {
//			totalTiles[i] --;
//			int key = std::get<0>(AgariChecker::GetInst()->getKey(totalTiles));
//			if (AgariChecker::GetInst()->isAgari(key)) {
//				totalTiles[tile] --;
//				totalTiles[i] ++;
//				return true;
//			} else {
//				totalTiles[i] ++;
//			}
//		}
//	}

	totalTiles[tile] --;
	return false;
}

namespace
{
bool mayChow(int t0, int t1) {
	if (t0 >= 27) {
		return false;
	}
	if (t0 / 9 != t1 / 9) {
		return false;
	}
	if ((t1 - t0) > 2) {
		return false;
	}
	return true;
}

bool mayChow(int t0, int t1, int t2) {
	if (t2 >= 27 || t1 >= 27 || t0 >= 27) {
		return false;
	}
	if ((t0 / 9 != t1 / 9) || (t1 / 9 != t2 / 9)) {
		return false;
	}
	if ((t1 != t0 + 1) || (t2 != t1 + 1)) {
		return false;
	}

	return true;
}

bool checkRemain(const vector<int>& remains, bool hasPair) {
	bool remainOk = true;
	if (remains.size() == 2) {
		if (hasPair) {
			if (remains[0] == remains[1]) {
				return true;
			} else if (mayChow(remains[0], remains[1])) {
				return true;
			} else {
				return false;
			}
		} else {
			//OK: potential pair
			return true;
		}
	} else if (remains.size() == 3) {
		if ((remains[0] == remains[1]) || (remains[1] == remains[2])) {
			return true;
		} else if (mayChow(remains[0], remains[1]) || mayChow(remains[1], remains[2])) {
			return true;
		} else {
			return false;
		}
	} else if (remains.size() > 3) {
		return false;
	} else { //remains <= 1
		return true;
	}
}

bool checkLastRemain(const vector<int>& remains, bool hasPair) {
	if (remains.size() < 2) { //Should have agari or single hang
		return false;
	} else if (remains.size() == 2) {
		if (!hasPair) {
			return true;
		} else {
			return false;
		}
	} else if (remains.size() == 3) {
		if ((remains[0] == remains[1]) || (remains[1] == remains[2])) {
			return true;
		} else if (mayChow(remains[0], remains[1]) || mayChow(remains[1], remains[2])) {
			return true;
		} else {
			return false;
		}
	} else {
		return false;
	}
}

//0, 0, 0, 3, 1, 2, 0, 1, 0, 0, 0, 3, 0, 0, 1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
bool checkRegular(vector<int>& tiles, vector<int>& remains, int startIndex, bool hasPair) {
//	printVec("checkRegular", tiles);
	for (; (tiles[startIndex] <= 0) && (startIndex < tiles.size()); startIndex++ );

	if (startIndex >= tiles.size()) {
//		return true;
		return checkLastRemain(remains, hasPair);
	}

	if (tiles[startIndex] >= 4) {
		tiles[startIndex] -= 4;
		if (checkRegular(tiles, remains, startIndex, hasPair)) {
			return true;
		}
		tiles[startIndex] += 4;
	}
	if (tiles[startIndex] >= 3) {
		tiles[startIndex] -= 3;
		if (checkRegular(tiles, remains, startIndex, hasPair)) {
			return true;
		}
		tiles[startIndex] += 3;
	}
	if (!hasPair && tiles[startIndex] >= 2) {
		tiles[startIndex] -= 2;
		if (checkRegular(tiles, remains, startIndex, true)) {
			return true;
		}
		tiles[startIndex] += 2;
	}
	if (startIndex <= 24) {
		if (tiles[startIndex] > 0 && tiles[startIndex + 1] > 0 && tiles[startIndex + 2] > 0) {
			if (mayChow(startIndex, startIndex + 1, startIndex + 2)) {
				tiles[startIndex] --;
				tiles[startIndex + 1] --;
				tiles[startIndex + 2] --;
				if (checkRegular(tiles, remains, startIndex, hasPair)) {
					return true;
				}
				tiles[startIndex] ++;
				tiles[startIndex + 1] ++;
				tiles[startIndex + 2] ++;
			}
		}
	}

	//TODO: tiralNum to be checked
	int tileNum = tiles[startIndex];
	//Above blocks has tried trialNum - 4 by default, 3 by pong trial, 2 by hasPair
	int trialNum = std::min(tileNum, hasPair? 2: 1);
	int origSize = remains.size();

	for (int i = 0; i < trialNum; i ++) {
		tiles[startIndex] --;
		remains.push_back(startIndex);
//		printVec("Check remain tiles: ", tiles);
//		printVec("Check remains: ", remains);

//		bool remainOk = true;
//		if (remains.size() == 2) {
//			if (hasPair) {
//				if (remains[0] == remains[1]) {
//					//OK, pong
//				} else if (mayChow(remains[0], remains[1])) {
//					//OK chow
//				} else {
//					remainOk = false;
//				}
//			} else {
//				//OK: potential pair
//			}
//		} else if (remains.size() == 3) {
//			if ((remains[0] == remains[1]) || (remains[1] == remains[2])) {
//				//continue;
//			} else if (mayChow(remains[0], remains[1]) || mayChow(remains[1], remains[2])) {
//				//continue;
//			} else {
//				remainOk = false;
//			}
//		} else if (remains.size() > 3) {
//			remainOk = false;
//		}
//
		bool remainOk = checkRemain(remains, hasPair);
		if (remainOk) {
			if (checkRegular(tiles, remains, startIndex, hasPair)) {
				return true;
			}
		}
	}

	remains.resize(origSize);
	tiles[startIndex] = tileNum;
	return false;
}
}

bool PlayerState::checkReach() {
	if (!closed) {
		return false;
	}
	if (reached) {
		logger->debug("Player{}:{} reached, no more reach", roomIndex, myIndex);
		return false;
	}
	logger->debug("Player {}:{} check reach", roomIndex, myIndex);
	printVec("Check reach", closeTiles);

	//7pair
	int singleCount = 0;
	for (int i = 0; i < closeTiles.size(); i ++) {
		if (closeTiles[i] % 2 != 0) {
			singleCount ++;
		}
	}
	if (singleCount == 2) {
		return true;
	}

	vector<int> reachTiles(closeTiles.begin(), closeTiles.end());
	vector<int> remains;
	return checkRegular(reachTiles, remains, 0, false);
}

bool PlayerState::checkChi(int tile) {
	if (tile >= 27) {
		return false;
	}

	if ((tile % 9 >= 2) && (closeTiles[tile - 1] > 0) && (closeTiles[tile - 2] > 0)) {
		return true;
	}

	if ((tile % 9 >= 1) && (tile % 9 <= 7) && (closeTiles[tile - 1] > 0) && (closeTiles[tile + 1] > 0)) {
		return true;
	}

	if ((tile % 9 <= 6) && (closeTiles[tile + 1] > 0) && (closeTiles[tile + 2] > 0)) {
		return true;
	}

	return false;
}

bool PlayerState::checkPong(int tile) {
	if (closeTiles[tile] >= 2) {
		return true;
	}

	return false;
}

//TODO: All kinds of kan
bool PlayerState::checkKan(int tile) {
	if (closeTiles[tile] == 3) {
		return true;
	}

//	for (int i = 0; i < meldTiles.size(); i ++) {
//		if ((meldTiles[i][0] == tile) && (meldTiles[i][1] == tile)) {
//			return true;
//		}
//	}

	return false;
}

int PlayerState::checkMeldType(int dropClientIndex, int tile) {
	if (reached) {
		logger->debug("Player{}:{} reached, no meld", roomIndex, myIndex);
		return -1;
	}

	bool canChow = false;
	bool canPong = false;
	bool canKan = false;

	if ((dropClientIndex + 1) % 4 == myIndex) {
		canChow = checkChi(tile);
	}
	canPong = checkPong(tile);
	canKan = checkKan(tile);

	if (canChow && canPong && canKan) {
		return 7;
	} else if (canPong && canKan) {
		return 3;
	} else if (canPong && canChow) {
		return 5;
	} else if (canPong) {
		return 1;
	} else if (canChow) {
		return 4;
	} else if (canKan) {
		return 2;
	}

	return -1;
}

void PlayerState::meldRaws(int raw, std::vector<int>& myRaws){
	logger->debug("State{}:{} meld raws {}", roomIndex, myIndex, raw);
//	logger->debug("raws: {}", myRaws);
	printVec("myRaws", myRaws);

	totalTiles[raw / 4] ++;

	if (myRaws.size() < 3) {
		for (int i = 0; i < myRaws.size(); i ++) {
			closeTiles[myRaws[i] / 4] --;
		}
		vector<int> melds(myRaws.begin(), myRaws.end());
		melds.push_back(raw);
		std::sort(melds.begin(), melds.end());
		meldTiles.push_back(melds);

		closed = false;
	} else if (myRaws.size() == 4){ //kan, raw from self
		int tile = raw / 4;
		if (closeTiles[tile] > 3) { //closed quad
			closeTiles[tile] = 0;
			vector<int> melds(myRaws.begin(), myRaws.end());
			std::sort(melds.begin(), melds.end());
			meldTiles.push_back(melds);
		} else { //Added open kan
			for (int i = 0; i < meldTiles.size(); i ++) {
				if ((meldTiles[i][0] / 4) == tile) {
					meldTiles[i].push_back(raw);
					break;
				}
			}
		}
	} else { //open kan
		int tile = raw / 4;
		for (int i = 0; i < meldTiles.size(); i ++) {
			if ((meldTiles[i][0] / 4) == tile) {
				meldTiles[i].push_back(raw);
				break;
			}
		}
	}
}


//bool PlayerState::IsHonor(int tile) {
//	return (tile >= 27);
//}
//
//bool PlayerState::IsTerminal(int tile) {
//	int remain = tile % 9;
//	return ((remain == 0) || (remain == 8));
//}
//
//bool PlayerState::IsTermOrHonor(int tile) {
//	return (IsHonor(tile) || IsTerminal(tile));
//}

//bool PlayerState::IsSeatWind(int tile) {
//	return ((tile >= 27) && (tile < 30));
//}
//
//bool PlayerState::IsKezi(vector<int>& tiles) {
//	if (tiles.size() > 2) {
//		if (tiles[0] == tiles[1]) {
//			return true;
//		}
//	}
//
//	return false;
//}

//bool PlayerState::IsShunzi(vector<int>& tiles) {
//	return !IsKezi(tiles);
//}

bool PlayerState::Is7Pair(vector<int>& tiles) {
	for (auto tile: tiles) {
		if (!((tile == 2) || (tile == 0))) {
			return false;
		}
	}

	return true;
}

//int PlayerState::calcMeldFu() {
//	int waitFu = calcWaitFu();
//	bool waitFuChecked = (waitFu >= 0);
//	int agariTile = agariRaw / 4;
//
//	int maxFu = 0;
//	//Close meld
//	for (int i = 0; i < combs.size(); i ++) {
//		int fu = 0;
//		int r = combs[i];
//		int numKezi = r & 0x07;
//		int numSeq = (r >> 3) & 0x07;
//		for (int j = 0; j < numKezi; j ++) {
//			int tile = poses[(r >> (10 + j * 4)) & 0x0F];
//			bool closeKezi = true;
//
//			for (int m = 0; m < meldTiles.size(); m ++) {
//				if (IsKezi(meldTiles[m]) && (meldTiles[m][0] / 4 == tile)) { //open kezi
//					if (meldTiles[m].size() > 3) { //kan
//						fu += Minkan;
//						if (IsTermOrHonor(tile)) {
//							fu += Minkan;
//						}
//					} else {
//						fu += Minko;
//						if (IsTermOrHonor(tile)) {
//							fu += Minko;
//						}
//					}
//					closeKezi = false;
//					break;
//				}
//			}
//
//			if (closeKezi) {
//				if (totalTiles[tile] == closeTiles[tile]) { //no extra tile for kan
//					fu += Ankor;
//					if (IsTermOrHonor(tile)) {
//						fu += Ankor;
//					}
//				} else {
//					fu += Ankan;
//					if (IsTermOrHonor(tile)) {
//						fu += Ankan;
//					}
//				}
//			}
//		}
//
//		for (int j = 0; j < numSeq; j ++) {
//			int tile = poses[(r >> (10 + numSeq * 4 + j * 4)) & 0x0F];
//			bool closeSeq = true;
//
//			if (!waitFuChecked) {
//				if ((tile + 1) == agariTile) {
//					waitFuChecked = true;
//					waitFu = Kanchan;
//				}
//			}
//			if (!waitFuChecked) {
//				if ((tile == agariTile) && (tile % 9 == 6)) {
//					waitFuChecked = true;
//					waitFu = Penchan;
//				}
//				if ((tile + 2 == agariTile) && (agariTile % 9 == 2)) {
//					waitFuChecked = true;
//					waitFu = Penchan;
//				}
//			}
//		}
//
//		int honorTile = poses[(r >> 6) & 0x0F];
//		if (!waitFuChecked) {
//			if (honorTile == agariTile) {
//				waitFu = Tanki;
//				waitFuChecked = true;
//			}
//		}
//		// no fu for pair when no wind considered
//
//		if (waitFuChecked) {
//			fu += waitFu;
//		}
//		if (fu > maxFu) {
//			maxFu = fu;
//		}
//	}
//
//	return maxFu;
//}
//
//int PlayerState::calcWaitFu() {
//	int fu = -1;
//	for (int i = 0; i < meldTiles.size(); i ++) {
//		for (int j = 0; j < meldTiles[i].size(); j ++) {
//			if (meldTiles[i][j] == agariRaw) {
//				if (meldTiles[i].size() < 3) { //pair
//					fu += Tanki;
//					return fu;
//				}
//
//				if (j == 1) {
//					fu += Kanchan;
//				}
//
//				int tile = agariRaw / 4;
//				if ((tile < 27) && (tile % 9 == 3 || tile % 9 == 7)) {
//					fu += Penchan;
//				}
//				return 0;
//			}
//		}
//	}
//
//	return fu;
//}
//
//int PlayerState::calcFu() {
//	//https://en.wikipedia.org/wiki/Japanese_Mahjong_scoring_rules#Counting_han
//	//6.
//	if (Is7Pair(totalTiles)) {
//		return Chit;
//	}
//	//1.
//	int fu = Futei;
//	//2.
//	if (closed && (fromWho != 0)) {
//		fu += Menzen;
//	}
//	//3.4
//	int meldFu = calcMeldFu();
//	fu += meldFu;
//	//5
//	if (closed && (fromWho == 0)) {
//		if (meldFu > 0) {
//			fu += Tsumo;
//		}
//	}
//	//7. TODO
//
//	return fu;
//}

int PlayerState::calcReward() {
	//Agari: winner 10, others -10;
	//Ryu: to be calculated
	return 0;
}

int PlayerState::GetChowM(int raw, std::vector<int> meldRaws) {
	int m = 0;

	meldRaws.push_back(raw);
	std::sort(meldRaws.begin(), meldRaws.end());
	int tile = meldRaws[0] / 4;
	logger->debug("GetChowM tile = {}", tile);

	int r = 0;
	for (int i = 0; i < 3; i ++) {
		if (meldRaws[i] == raw) {
			r = i;
		}
		meldRaws[i] %= 4;
		m |= (meldRaws[i] << (3 + i * 2));
	}
	logger->debug("GetChowM r = {} m = {}", r, m);


	int squeezeTile = tile / 9 * 7 + (tile % 9);
	int tileValue = squeezeTile * 3 + r;

	m |= (tileValue << 10);
	m |= 0b111;

	return m;
}

int PlayerState::GetPongM(int raw, std::vector<int> meldRaws) {
	int m = 0;
	int tile = raw / 4;

	meldRaws.push_back(raw);
	std::sort(meldRaws.begin(), meldRaws.end());

	int r = 0;
	int missed = -1;
	for (int i = 0; i < meldRaws.size(); i ++) {
		if (meldRaws[i] == raw) {
			r = i;
		}

		if ((tile * 4 + i) == meldRaws[i]) {
			//nothing
		} else {
			if (missed < 0) {
				missed = i;
			}
		}
	}
	if (missed < 0) {
		missed = 3;
	}

	if (meldRaws.size() == 3) {
		int tileValue = tile * 3 + r;

		m |= (tileValue << 9); //tile and meld tile pos in vec
		m |= (missed << 5); //missed tile
		m |= 0b010; //last 3 bits
		m |= 0x08; //pong indicator
	} else {
		int tileValue = tile * 3 + r;
		m |= 0b010;
		m |= 0x10;
		//no missing
	}

	return m;
}

int PlayerState::GetKanM(int raw, std::vector<int>& meldRaws) {
	int m = 0;
	int tile = raw / 4;
	int called = -1;

	if (meldRaws.size() == 3) {
		for (int i = 0; i < meldRaws.size(); i ++) {
			if ((tile * 4 + i) != meldRaws[i]) {
				called = i;
				break;
			}
		}
		if (called < 0) {
			called = 3;
		}
	} else { //kan from me
		for (int i = 0; i < meldRaws.size(); i ++) {
			if (tile * 4 + i == raw) {
				called = i;
				break;
			}
		}
	}

	int baseCalled = tile * 4 + called;
	m |= (baseCalled << 8);
	m &= ~(0x04 | 0x18 | 0x20);

	return m;
}

int PlayerState::GetM(int type, int raw, std::vector<int>& meldRaws) {
	logger->debug("GetM type = {}, raw = {}", type, raw);
	printVec("raws: ", meldRaws);
	if (meldRaws.size() > 2) {
		return GetKanM(raw, meldRaws);
	} else {
		if (type == 3) {
			return GetChowM(raw, meldRaws);
		} else {
			return GetPongM(raw, meldRaws);
		}
	}
}
