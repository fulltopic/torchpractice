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

class Room;

class PlayerState {
private:
	std::vector<std::vector<int>> meldTiles; //raws
	std::vector<int> closeTiles; //counts
	std::vector<int> dropTiles;
	std::vector<int> totalTiles;
	std::vector<int> acceptRaws;

	std::vector<int> combs;
	std::vector<int> poses;

	const int myIndex;
	const int roomIndex;
	bool closed;
	bool reached;
	int agariRaw;
	int fromWho;

//	int calcMeldFu();
//	int calcWaitFu();
//	int calcFu();
public:
	enum Fu {
		Futei = 20,
		Menzen = 10,
		Chitoitsu = 50,
		OpenPinfu = 30,
		Mangan = 2000, //TODO: Maybe 200?
		Tsumo = 2,
		Chit = 50,
	};
	enum MeldFu {
		Minko = 2,
		Ankor = 4,
		Minkan = 8,
		Ankan = 16,
		Shuntsu = 0,
		Toitsu = 0,
	};
	enum WaitFu {
		Ryanmen = 0,
		Kanchan = 2,
		Penchan = 2,
		Tanki = 2,
		Shanpon = 0,
	};


	PlayerState(int index, int rIndex);
	void reset();

	void acceptTile (int raw);
	void dropTile (int raw);
//	void meldTile (int m);
	bool checkAgari (int raw);
	bool checkReach();
	int calcReward();
	bool checkChi(int tile);
	bool checkPong(int tile);
	bool checkKan(int tile);
	int checkMeldType (int dropClientIndex, int tile);
	void meldRaws(int raw, std::vector<int>& myRaws);

//	static bool IsTerminal(int tile);
//	static bool IsHonor(int tile);
//	static bool IsTermOrHonor(int tile);
//	static bool IsSeatWind(int tile);
//	static bool IsKezi(std::vector<int>& tiles);
//	static bool IsShunzi(std::vector<int>& tiles);
	static bool Is7Pair(std::vector<int>& tiles);

	static int GetChowM(int raw, std::vector<int> meldRaws);
	static int GetPongM(int raw, std::vector<int> meldRaws);
	static int GetKanM(int raw, std::vector<int>& meldRaws);
	static int GetM(int type, int raw, std::vector<int>& meldRaws);

	friend class Room;
};



#endif /* INCLUDE_SELFSERVER_PLAYERSTATE_H_ */
