/*
 * playerstate.h
 *
 *  Created on: Oct 31, 2020
 *      Author: zf
 */

#ifndef INCLUDE_SELFSERVER_PLAYERSTATE_H_
#define INCLUDE_SELFSERVER_PLAYERSTATE_H_

#include "agarichecker.h"

#include <vector>
#include <map>

enum class MeldType: int {
	Reach = 32,
	Tsumo = 16,
	RonByDrop = 9,
	ChowPongKan = 7,
	PongChow = 5,
	PongKan = 3,
	Chow = 4,
	Kan = 2,
	Pong = 1,

	RonRspInd6 = 6,
	RonRspInd7 = 7,
	RonRspInd9 = 9,

	ChowRspType = 3,
	PongRspType = 1,

	Invalid = -1,
};

class MeldTypeHelper {
public:
	static bool IsValid(const MeldType& meldType);
	static bool Prior2(const MeldType type0, const MeldType type1);
	static bool IsReachInd(const MeldType& type);
	static bool IsRonInd(const MeldType& type);
	static bool IsKanRsp(const MeldType& type);
};


class Room;

class PlayerState {
private:
	std::vector<std::vector<int>> meldTiles; //raws
	std::vector<int> closeTiles = std::vector<int>(34, 0); //counts
	std::vector<int> dropTiles = std::vector<int>(34,  0);
	std::vector<int> totalTiles = std::vector<int>(34, 0);
	std::vector<int> acceptRaws;

	std::vector<int> combs;
	std::vector<int> poses;

	bool closed = true;
	bool reached = false;
	int agariRaw = -1;
	int fromWho = -1;

	const int myIndex;
	const int roomIndex;

	class KanTileObj {
	private:
		int& counter;
	public:
		KanTileObj(int& inCounter): counter(inCounter) {
			counter ++;
		}

		~KanTileObj() {
			counter --;
		}

		KanTileObj(const KanTileObj&) = delete;
		KanTileObj& operator=(const KanTileObj&) = delete;
		KanTileObj(const KanTileObj&&) = delete;
		KanTileObj& operator=(const KanTileObj&&) = delete;
	};
//	int calcMeldFu();
//	int calcWaitFu();
//	int calcFu();
public:
//	enum Fu {
//		Futei = 20,
//		Menzen = 10,
//		Chitoitsu = 50,
//		OpenPinfu = 30,
//		Mangan = 2000, //TODO: Maybe 200?
//		Tsumo = 2,
//		Chit = 50,
//	};
//	enum MeldFu {
//		Minko = 2,
//		Ankor = 4,
//		Minkan = 8,
//		Ankan = 16,
//		Shuntsu = 0,
//		Toitsu = 0,
//	};
//	enum WaitFu {
//		Ryanmen = 0,
//		Kanchan = 2,
//		Penchan = 2,
//		Tanki = 2,
//		Shanpon = 0,
//	};

//	static bool IsTerminal(int tile);
//	static bool IsHonor(int tile);
//	static bool IsTermOrHonor(int tile);
//	static bool IsSeatWind(int tile);
//	static bool IsKezi(std::vector<int>& tiles);
//	static bool IsShunzi(std::vector<int>& tiles);

	PlayerState(int index, int rIndex);
	void reset();

	void acceptTile (int raw);
	void dropTile (int raw);
	bool checkAgari (int raw);
	bool checkReach();
	int calcReward();
	bool checkChi(int tile);
	bool checkPong(int tile);
	bool checkKan(int tile);
	MeldType checkMeldType (int dropClientIndex, int tile);
	void meldRaws(int raw, const std::vector<int>& myRaws);


	static bool Is7Pair(std::vector<int>& tiles);

	static int GetChowM(int raw, const std::vector<int>& inputRaws);
	static int GetPongM(int raw, const std::vector<int>& inputRaws);
	static int GetKanM(int raw, const std::vector<int>& meldRaws);
	static int GetM(MeldType type, int raw, const std::vector<int>& meldRaws);

	friend class Room;
};



#endif /* INCLUDE_SELFSERVER_PLAYERSTATE_H_ */
