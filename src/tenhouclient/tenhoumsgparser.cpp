/*
 * tenhoumsgparser.cpp
 *
 *  Created on: Apr 10, 2020
 *      Author: zf
 */

#include "tenhouclient/tenhouconsts.h"
#include "tenhouclient/tenhoumsgparser.h"
#include "tenhouclient/logger.h"

#include <boost/algorithm/string.hpp>
#include <iostream>
#include <string>
#include <set>

#include <boost/xpressive/xpressive.hpp>


using namespace boost;
using namespace boost::xpressive;
using namespace std;

const std::set<string> TenhouMsgParser::Keys {"TAIKYOKU", " INIT", "T", "D", "U", "E", "V", "F", "W", "G", "N", "RYUUKYOKU", "AGARI", "REACH", "DORA"};
const std::set<string> TenhouMsgParser::SceneKeys {"HELO", "JOIN", "REJOIN", "UN", "LN", "GO", "PROF"};

std::string TenhouMsgParser::RemoveWrapper (std::string msg) {
	replace_all(msg, "<", " ");
	replace_all(msg, "/>", " ");
	trim(msg);

	return msg;
}

string TenhouMsgParser::RemoveHead (const std::string head, std::string msg) {
	replace_all(msg, head, " ");
	replace_all(msg, "\"", " ");
	trim(msg);

	return msg;
}

vector<string> TenhouMsgParser::ParseValues(string msg, const string head, const string token) {
	static auto logger = Logger::GetLogger();

	string valueMsg = RemoveHead(head, msg);
	logger->debug("Message after head: {}", valueMsg);
	vector<string> items;
	split(items, valueMsg, is_any_of(token), token_compress_on);
	logger->debug("Get value items {}", items.size());
	return items;
}

vector<string> TenhouMsgParser::ParseItems(string msg) {
	msg = RemoveWrapper(msg);

	vector<string> items;
	split(items, msg, is_any_of(" "), token_compress_on);

	return items;
}

vector<string> TenhouMsgParser::ParseHeloReply(string msg) {
	vector<string> items = ParseItems(msg);

	for (int i = 0; i < items.size(); i ++) {
		if (items[i].find("auth") != string::npos) {
			string authMsg = items[i];
			trim(authMsg);
			authMsg = RemoveHead("auth=\"", authMsg);

			vector<string> parts;
			split(parts, authMsg, is_any_of("-"), token_compress_on);

			return parts;
		}
	}

	return vector<string>();
}



int TenhouMsgParser::Raw2Tile (int rawTile) {
	return rawTile / NumPerTile;
}

int TenhouMsgParser::Key2PlayerIndex (string key) {
	static map<string, int> keyMap {
		{"t", 0},
		{"u", 1},
		{"v", 2},
		{"w", 3},
		{"R", 0},
		{"U", 1},
		{"V", 2},
		{"W", 3},
		{"d", 0},
		{"e", 1},
		{"f", 2},
		{"g", 3},
		{"D", 0},
		{"E", 1},
		{"F", 2},
		{"G", 3},
	};

	return keyMap[key];
}

string TenhouMsgParser::Index2PlayerDropKey (int playerIndex) {
	static map<int, string> indexMap {
		{0, "D"},
		{1, "E"},
		{2, "F"},
		{3, "G"}
	};

	return indexMap[playerIndex];
}

InitResult TenhouMsgParser::ParseInit (string msg) {
	static auto logger = Logger::GetLogger();

//	logger->debug("To parse init msg ");
	msg = RemoveWrapper(msg);
//	logger->debug("Removed wrapper");

	vector<string> items;
	split (items, msg, is_any_of(" "), token_compress_on);

	int oya = 0;
	vector<int> tiles;

	for (int i = 0; i < items.size(); i ++) {
//		logger->debug("Parse item {}", items[i]);
		if (items[i].find("oya") != string::npos) {
//			logger->debug("Parse oya: {}", items[i]);
			string oyaMsg = items[i];
			oya = stoi(RemoveHead("oya=\"", oyaMsg));
			logger->debug("Get oya {}", oya);
		}

		if (items[i].find("hai") != string::npos) {
//			logger->debug("Parse hais {}", items[i]);
			string haiMsg = items[i];
			trim(haiMsg);
			auto values = ParseValues(haiMsg, "hai=\"", ",");
			//TODO: To assign values (size(), 0)
			for (int j = 0; j < values.size(); j ++) {
//				logger->debug("stoi {}", values[j]);
				tiles.push_back(stoi(values[j]));
			}
//		string stateMsg = items[4 + i];
//		trim(stateMsg);
//		erase_head(stateMsg, 6);
//		erase_all(stateMsg, "\"");
//		trim(stateMsg);
//
//		vector<string> tiles;
//		split(tiles, stateMsg, is_any_of(","), token_compress_on);
//		for (int j = 0; j < UsualTileNum; j ++) {
//			states[i][j] = atoi(tiles[j].c_str());
//		}

		}
	}
	return {oya, tiles};
}

DropResult TenhouMsgParser::ParseDrop (std::string msg) {
	msg = RemoveWrapper(msg);
	string playerMsg = msg.substr(0, 1);
	int playerIndex = Key2PlayerIndex(playerMsg);

	string rawMsg = msg.substr(1);
	int rawTile = atoi(rawMsg.c_str());

	return {playerIndex, rawTile};
}

AcceptResult TenhouMsgParser::ParseAccept (std::string msg) {
	msg = RemoveWrapper(msg);
	string rawMsg = msg.substr(1);
	int rawTile = atoi(rawMsg.c_str());

	return rawTile;
}

int TenhouMsgParser::ParseHead(const std::string head, const std::string msg) {
//	cout << "Input msg " << msg << endl;
	string valueMsg = msg.substr(head.size(), msg.size() - head.size());
//	cout << "Remove head " << valueMsg << endl;
	valueMsg = valueMsg.substr(0, valueMsg.size() - 1);
//	cout << "Remove tail " << valueMsg << endl;
	return atoi(valueMsg.c_str());
}

int TenhouMsgParser::ParseWho(std::string msg) {
	static string WhoHead("who=\"");
	return ParseHead(WhoHead, msg);
}

int TenhouMsgParser::ParseM(std::string msg) {
	static string MHead("m=\"");
	return ParseHead(MHead, msg);
}

//TODO: Furiten

StealResult TenhouMsgParser::ParseSteal(std::string msg) {
	msg = RemoveWrapper(msg);
	vector<string> items;
	split(items, msg, is_any_of(" "), token_compress_on);

	int who = ParseWho(items[1]);
	int m = ParseM(items[2]);
	Logger::GetLogger()->debug("Pass N msg for {}", m);
//	cout << "Get m " << m << " from " << items[2] << endl;



	if ((m & 0x4) > 0) {
		return ParseChow(who, m);
	} else if ((m & 0x18) > 0) {
		return ParsePong(who, m);
	} else if ((m & 0x20) > 0) {
		Logger::GetLogger()->error("Received nuki, don't know how to deal with");
		return {KitaBits, -1, -1, {0, 0, 0}};
	} else {
		return ParseKan(who, m);
	}




//	else if ((m & KakanFlag) > 0) {
//		Logger::GetLogger()->debug("Get kakan flag");
//		return ParseKankan(who, m);
//	}else if ((m & AnkanFlag) == 0) {
//		Logger::GetLogger()->debug("Parse N for Ankan {}", who);
//		return ParseAnkan(who, m);
//	} else if (((m >> KitaBits) & 1) == 1) {
//		return {KitaBits, -1, -1, {0, 0, 0}};
//	} else {
//		Logger::GetLogger()->warn("Minkan, Maybe invalid flag: {}", m);
//		return ParseMinkan(who, m);
//	}


//	if ((m & ChowFlag) > 0) {
//		return ParseChow(who, m);
//	} else if ((m & PongFlag) > 0) {
//		return ParsePong(who, m);
//	} else if ((m & KakanFlag) > 0) {
//		Logger::GetLogger()->debug("Get kakan flag");
//		return ParseKankan(who, m);
//	}else if ((m & AnkanFlag) == 0) {
//		Logger::GetLogger()->debug("Parse N for Ankan {}", who);
//		return ParseAnkan(who, m);
//	} else if (((m >> KitaBits) & 1) == 1) {
//		return {KitaBits, -1, -1, {0, 0, 0}};
//	} else {
//		Logger::GetLogger()->warn("Minkan, Maybe invalid flag: {}", m);
//		return ParseMinkan(who, m);
//	}

//	return {InvalidFlag, -1, -1, {0, 0, 0}};
}

StealResult TenhouMsgParser::ParseChow (int who, int m) {
	int chowTile = (m >> 10) & 63;
	int r = chowTile % 3;
	Logger::GetLogger()->debug("ParseChow {}, {}", chowTile, r);

	chowTile /= 3;
	chowTile = chowTile / 7 * 9 + chowTile % 7;
	chowTile *= 4;
	Logger::GetLogger()->debug("Chow tile {}", chowTile);

	vector<int> candidates(3, 0);
	candidates[0] = chowTile + ((m >> 3) & 3);
	candidates[1] = chowTile + 4 + ((m >> 5) & 3);
	candidates[2] = chowTile + 8 + ((m >> 7) & 3);
	Logger::GetLogger()->debug("Candidates: {} {} {}", candidates[0], candidates[1], candidates[2]);

	vector<int> peers(3, 0);
	switch(r) {
	case 0:
		for (int i = 0; i < 3; i ++) {
			peers[i] = candidates[i];
		}
		break;
	case 1:
		peers[0] = candidates[1];
		peers[1] = candidates[0];
		peers[2] = candidates[2];
		break;
	case 2:
		peers[0] = candidates[2];
		peers[1] = candidates[0];
		peers[2] = candidates[1];
	}

	return {
		ChowFlag,
		who,
		candidates[r],
		peers
	};
}

StealResult TenhouMsgParser::ParsePong (int who, int m) {
	static const vector<int> Poses {0, 1, 2, 3};

	int t4 = (m >> 5) & 3;
	vector<int> pongPoses;
	for (int i = 0; i < NumPerTile; i ++) {
		if (i != t4) {
			pongPoses.push_back(i);
		}
	}

	int baseCalled = m >> 9;
	int base = baseCalled / 3;
	int called = baseCalled % 3;

	if ((m & 0x8) > 0) {
		vector<int> tiles {pongPoses[0] + base * NumPerTile, pongPoses[1] + base * NumPerTile, pongPoses[2] + base * NumPerTile};
		for (int i = 0; i < tiles.size(); i ++) {
			if (pongPoses[i] == called) {
				swap(tiles[i], tiles[0]);
			}
		}


		Logger::GetLogger()->warn("Get pong tiles {}, {}, {}", tiles[0], tiles[1], tiles[2]);
		return {PongFlag, who, called + base * NumPerTile, tiles};
	} else {
		vector<int> tiles { base * NumPerTile, 1 + base * NumPerTile, 2 + base * NumPerTile, 3 + base * NumPerTile};
		swap(tiles[0], tiles[called]);

		Logger::GetLogger()->warn("Get kakan tiles {}, {}, {}, {}", tiles[0], tiles[1], tiles[2], tiles[3]);
		return {KanFlag, who, called + base * NumPerTile, tiles};
	}
//    int unused = (m >> 5) & 3;
//    int tmp = (m >> 9) & 127;
//    int r = tmp % 3;
//
//    tmp /= 3;
//    tmp *= 4;
//
//    vector<int> selfHai(2, 0);
//    int count = 0;
//    int idx = 0;
//
//    cout << "unused " << unused << ", r " << r << endl;
//    for (int i = 0; i < NumPerTile; i ++) {
//      if (i != unused) {
//    	  count ++;
//        if (count != r) {
//          selfHai[idx] = tmp + i;
//          idx ++;
//        }
//      }
//    }
//
//    int pongTile = tmp + r;
//
//
//
//    Logger::GetLogger()->error("Get pong tiles {}, {}, {}", pongTile, selfHai[0], selfHai[1]);
//    return {PongFlag, who, pongTile, {pongTile, selfHai[0], selfHai[1]}};
}

StealResult TenhouMsgParser::ParseKankan(int who, int m) {

	int raw = m >> 8;
	int base = raw / 4;
	Logger::GetLogger()->debug("Parse Kakan for {}, get raw {}", m, raw);

	return {KakanFlag, who, raw, {base * NumPerTile, base * NumPerTile + 1, base * NumPerTile + 2, base * NumPerTile + 3}};

}

StealResult TenhouMsgParser::ParseAnkan(int who, int m) {
	int raw = (m >> 8) & 255;
	Logger::GetLogger()->debug("Ankan raw {}", raw);
	int tile = raw / 4 * 4;
	int rawIndex = raw - tile;

	vector<int> tiles(4, 0);
	for (int i = 0; i < NumPerTile; i ++) {
		tiles[i] = tile + i;
	}
	swap(tiles[0], tiles[rawIndex]);

	return {AnkanFlag, who, raw, tiles};
//	return {AnkanFlag, who, raw, {raw}}; //Only the melt tile to be fixed
	//TODO: To deal with raw
}

StealResult TenhouMsgParser::ParseMinkan(int who, int m) {
	int nakiHai = (m >> 8) & 255;
	int haiFirst = nakiHai / 4 * 4;
	vector<int> selfHai(4, 0);
	selfHai[0] = nakiHai;
	int index = 0;
	for (int i = 1; i < 4; i ++) {
		if ((haiFirst + index) == nakiHai) index ++;
		selfHai[i] = haiFirst = index;
		index ++;
	}

	return {MinkanFlag, who, nakiHai, selfHai};
}

StealResult TenhouMsgParser::ParseKan(int who, int m) {
	int raw = (m >> 8) & 255;
	Logger::GetLogger()->debug("Kan raw {}", raw);
	int tile = raw / 4 * 4;
	int rawIndex = raw - tile;

	vector<int> tiles(4, 0);
	for (int i = 0; i < NumPerTile; i ++) {
		tiles[i] = tile + i;
	}
	swap(tiles[0], tiles[rawIndex]);

	return {KanFlag, who, raw, tiles};
}

ReachResult TenhouMsgParser::ParseReach (std::string msg) {
	string reachMsg = RemoveWrapper(msg);
	vector<string> items;
	split(items, msg, is_any_of(" "), token_compress_on);
	int who = -1;
	int step = -1;

	for (int i = 0; i < items.size(); i ++) {
		if (items[i].find("who") != string::npos) {
			trim(items[i]);
			who = ParseHead("who=\"", items[i]);
		}

		if (items[i].find("step") != string::npos) {
			trim(items[i]);
			step = ParseHead("step=\"", items[i]);
		}
	}
	return {who, step};
}

AgariResult TenhouMsgParser::ParseAgari (std::string msg) {
	vector<string> items = ParseItems(msg);
	int who = -1;
	int machi = -1;
	int reward = 0;

	for (int i = 0; i < items.size(); i ++) {
//		cout << "Parse item " << items[i] << endl;
		if (items[i].find("who") != string::npos) {
			who = ParseHead("who=\"", items[i]);
		}
		if (items[i].find("machi") != string::npos) {
			machi = ParseHead("machi=\"", items[i]);
		}
		if (items[i].find("sc") != string::npos) {
			string valueMsg = RemoveHead("sc=\"", items[i]);
//			cout << "Value msg " << valueMsg << endl;
			vector<string> valueItems = ParseValues(valueMsg, "sc=\"", ",");
//			for (int j = 0; j < valueItems.size(); j ++) {
//				cout << valueItems[j] << ", ";
//			}
//			cout << endl;
			reward = atoi(valueItems[1].c_str());
		}
	}

	return {who, reward};
}

//TODO: To fill from who
StealIndicator TenhouMsgParser::ParseStealIndicator(std::string msg) {
	vector<string> items = ParseItems(msg);

	string tileMsg = items[0];
	trim(tileMsg);

	string whoMsg = tileMsg.substr(0, 1);
	int fromWho = Key2PlayerIndex(whoMsg);

	tileMsg = tileMsg.substr(1);
	int tile = atoi(tileMsg.c_str());

	string typeMsg = items[1];
	typeMsg = RemoveHead("t=\"", typeMsg);
	trim(typeMsg);
	int type = atoi(typeMsg.c_str());

	return {fromWho, tile, type};
}


RyuResult TenhouMsgParser::ParseRyu (std::string msg) {
	vector<string> items = ParseItems(msg);

	for (int i = 0; i < items.size(); i ++) {
		if (items[i].find("sc") != string::npos) {
			vector<string> values = ParseValues(items[i], "sc=\"", ",");
			return atoi(values[1].c_str());
		}
	}

	return 0;
}

int TenhouMsgParser::ParseDora(string msg) {
	auto items = ParseItems(msg);
	for (int i = 0; i < items.size(); i ++) {
		if (items[i].find("hai") != string::npos) {
			auto doraStr = RemoveHead("hai=\"", items[i]);
			return stoi(doraStr);
		}
	}

	return -1;
}

#define RxMatch	\
	smatch what;	\
	return regex_search(msg, what, rx);

bool TenhouMsgParser::IsValidMsg(string msg) {
	static vector<string> MsgHeads {
		"<AGARI",
		"<RYU",
		"<INIT",
		"<N",
		"<REACH",
		"<FURITEN",
		"<DORA",


		"<NEXT",
		"<Z",
		"<HELO",
		"<AUTH",
		"<LN",
		"<PXR",
		"<JOIN",
		"<REJOIN",
		"<GO",
		"<UN",
		"<TAIKYO",
		"<PROF",


		"<U",
		"<E",
		"<V",
		"<F",
		"<W",
		"<G",
		"<T",
		"<D",
		"<u",
		"<e",
		"<v",
		"<f",
		"<w",
		"<g",
		"<t",
		"<d",

		"TIMEOUT",
	};

	trim(msg);
	for (int i = 0; i < MsgHeads.size(); i ++) {
		if (msg.find(MsgHeads[i]) != string::npos) {
			return true;
		}
	}

	return false;
}

bool TenhouMsgParser::IsSceneEnd(string msg) {
	return (msg.find("BYE") != string::npos)
			|| (msg.find("PROF") != string::npos);
}

bool TenhouMsgParser::IsGameEnd(string msg) {
	return (msg.find("AGARI") != string::npos)
			|| (msg.find("RYU") != string::npos);
}

bool TenhouMsgParser::IsDropMsg(const string msg) {
	static xpressive::sregex rx = xpressive::as_xpr('<')
		>> xpressive::icase(xpressive::as_xpr('d') | 'e' | 'f' | 'g');
//		>> +_d;

	RxMatch;
}

bool TenhouMsgParser::IsAcceptMsg(string msg) {
	static sregex rx = as_xpr('<') >> icase(as_xpr('t')) >> +_d;
//	smatch what;
//
//	return regex_search(msg, what, rx);
	RxMatch;
}

bool TenhouMsgParser::IsNMsg(string msg) {
	static sregex rx = as_xpr('<') >> as_xpr('N') >> as_xpr(' ') >> "who";

	RxMatch;
}

bool TenhouMsgParser::IsIndMsg(string msg) {
	sregex rx = as_xpr(' ') >> "t=\"" >> +_d;

	RxMatch;
}

bool TenhouMsgParser::IsSilentMsg(string msg) {
	sregex rx = as_xpr('<') >> icase(as_xpr('u') | 'v' | 'w');

	RxMatch;
}

bool TenhouMsgParser::IsDoraMsg(const string msg) {
	return (msg.find("<DORA") != string::npos);
}
GameMsgType TenhouMsgParser::GetMsgType(const string msg) {
	if (msg.find("INIT") != string::npos) {
		return GameMsgType::InitMsg;
	} else if (IsDoraMsg(msg)) {
		return GameMsgType::DoraMsg;
	} else if (IsIndMsg(msg)) {
		return GameMsgType::IndicatorMsg;
	} else if (IsDropMsg(msg)) {
		return GameMsgType::DropMsg;
	} else if (IsAcceptMsg(msg)) {
		return GameMsgType::AcceptMsg;
	} else if (IsNMsg(msg)) {
		return GameMsgType::NMsg;
	} else if (msg.find("REACH") != string::npos) {
		return GameMsgType::ReachMsg;
	} else if (IsGameEnd(msg)) {
		return GameMsgType::GameEndMsg;
	} else if (IsSceneEnd(msg)) {
		return GameMsgType::SceneEndMsg;
	} else if (IsSilentMsg(msg)) {
		return GameMsgType::SilentMsg;
	}

	else {
		return GameMsgType::InvalidMsg;
	}
}

StealType TenhouMsgParser::GetIndType(const string msg) {
	string stealMsg = msg;
	auto items = ParseItems(stealMsg);

	for (int i = 0; i < items.size(); i ++) {
		if (items[i].find("t=") != string::npos) {
			trim(items[i]);
			int t = stoi(RemoveHead("t=\"", items[i]));

			switch(t) {
			case 1:
				return StealType::PongType;
			case 2:
				return StealType::KanType;
			case 3:
				return StealType::PonKanType;
			case 4:
				return StealType::ChowType;
			case 5:
				return StealType::PonChowType;
			case 7:
				return StealType::PonChowKanType;
			case 8:
			case 9:
			case 10:
			case 11:
			case 12:
			case 13:
			case 15:
			case 16:
			case 48:
			case 64: // Don't know what
				return StealType::RonType;
			case 32:
				return StealType::ReachType;
			}
		}
	}

	return StealType::UnknownType;
}

bool TenhouMsgParser::IsTsumogiriMsg(const string msg) {
	string tMsg = msg;
	if ((msg.find("<e") != string::npos)
			|| (msg.find("<f") != string::npos)
			|| (msg.find("<g") != string::npos)) {
		return true;
	}

	return false;
}
