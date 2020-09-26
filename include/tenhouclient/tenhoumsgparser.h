/*
 * tenhoumsgparser.h
 *
 *  Created on: Apr 10, 2020
 *      Author: zf
 */

#ifndef INCLUDE_TENHOUCLIENT_TENHOUMSGPARSER_H_
#define INCLUDE_TENHOUCLIENT_TENHOUMSGPARSER_H_

#include <string>
#include <vector>
#include <set>
#include <map>

//All return raw tile
enum GameMsgType {
	InitMsg,
	DropMsg,
	AcceptMsg,
	NMsg,
	ReachMsg,
	IndicatorMsg,

	SilentMsg,

	GameEndMsg,
	SceneEndMsg,

	DoraMsg,

	REINITMsg,

	InvalidMsg,
};

enum StealType {
	PongType = 1,
	KanType = 2,
	PonKanType = 3,
	ChowType = 4,
	PonChowType = 5,
	PonChowKanType = 7,
	ReachType,
	RonType,

	DropType,
	UnknownType,
};

using AcceptResult = int;

struct DropResult {
	int playerIndex;
	int tile;
};

struct ChowResult {
	int playerIndex;
	int chowTile;
	std::vector<int> tiles;
};

struct PongResult {
	int playerIndex;
	std::vector<int> pongTile;
};

struct InitResult {
	int oyaIndex;
//	int ownerIndex;
	std::vector<int> tiles;
};

struct ReachResult {
	int playerIndex;
	int reachPhase;
};

struct AgariResult {
	int winnerIndex;
	int machi;
	int reward;
};

//struct StealResult {
//	int who;
//	int m;
//};

struct StealResult {
	int flag;
	int playerIndex;
	int stealTile;
	std::vector<int> tiles;
};

struct StealIndicator {
	int who;
	int tile;
	int type;
};

using RyuResult = int;

class TenhouMsgParser {
private:
	TenhouMsgParser() = delete;
	TenhouMsgParser(TenhouMsgParser&) = delete;
	TenhouMsgParser& operator=(TenhouMsgParser&) = delete;
	~TenhouMsgParser() = delete;

public:
	const static std::set<std::string> Keys;
	const static std::set<std::string> SceneKeys;

	static std::string RemoveWrapper (std::string msg);
	static std::string RemoveHead (const std::string head, std::string msg);
	static std::vector<std::string> ParseItems(std::string msg);
	static std::vector<std::string> ParseValues(std::string msg, const std::string head, const std::string token);

	static std::vector<std::string> ParseHeloReply(std::string msg);

	static int Raw2Tile (int rawTile) ;
	static int Key2PlayerIndex (std::string key);
	static std::string Index2PlayerDropKey (int playerIndex);

	static InitResult ParseInit (std::string msg);

	static DropResult ParseDrop (std::string msg);
	static AcceptResult ParseAccept (std::string msg);

	static StealResult ParseSteal(std::string msg);
	static int ParseHead(const std::string head, const std::string msg);
	static int ParseWho(std::string msg);
	static int ParseM(std::string msg);
	static StealResult ParseChow (int who, int m);
	static StealResult ParsePong (int who, int m);
	static StealResult ParseKankan(int who, int m);
	static StealResult ParseAnkan(int who, int m);
	static StealResult ParseMinkan(int who, int m);
	static StealResult ParseKan(int who, int m);
	static ReachResult ParseReach (std::string msg);
	static StealIndicator ParseStealIndicator(std::string msg);

	static AgariResult ParseAgari (std::string msg);
	static RyuResult ParseRyu (std::string msg);
	static int ParseDora (std::string msg);

	static bool IsValidMsg(std::string msg);
	static bool IsGameEnd(std::string msg);
	static bool IsSceneEnd(std::string msg);

	static bool IsDoraMsg(const std::string msg);
	static bool IsDropMsg(const std::string msg);
	static bool IsAcceptMsg(const std::string msg);
	static bool IsNMsg(const std::string msg);
	static bool IsIndMsg(const std::string msg);
	static bool IsSilentMsg(const std::string msg);
	static GameMsgType GetMsgType(const std::string msg);

	static StealType GetIndType(const std::string msg);
	static bool IsTsumogiriMsg(const std::string msg);

	static std::vector<int> ParseReinitItems(std::string msg, const std::string key);
	static std::vector<std::vector<int>> ParseReinitMsg (std::string msg);
	static std::vector<int> ParseReinitM (std::string msg); //"m2=\"43601\""
};



#endif /* INCLUDE_TENHOUCLIENT_TENHOUMSGPARSER_H_ */
