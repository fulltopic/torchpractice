/*
 * tilpatternutils.cpp
 *
 *  Created on: Oct 29, 2020
 *      Author: zf
 */



#include "selfserver/tilepatternutils.h"

#include <vector>
#include <string>
#include <iostream>
#include <sstream>

using S = std::string;
using std::vector;

int TilePatternUtils::GetMajorNum (const std::vector<int>& tiles) {
	int majorCount = 0;
	int lastMajor = -1;
	for (const auto tile: tiles) {
		if ((tile / TilePerMajor) != lastMajor) {
			majorCount ++;
			lastMajor = (tile / TilePerMajor);
		}
	}

	return lastMajor;
}

int TilePatternUtils::GetMsgIndex(int index, int myIndex) {
	return (index - myIndex + PlayerNum) % PlayerNum;
}

S TilePatternUtils::IntVec2Str(S head, const vector<int>& data) {
	if (data.size() == 0) {
		return "";
	}

	std::stringbuf buf;
	std::ostream output(&buf);

	output << head << "=\"" << data[0];
	for (int i = 1; i < data.size(); i ++) {
		output << "," << data[i];
	}
	output << "\"";

	return buf.str();
}

bool TilePatternUtils::IsKyushu(const vector<int>& tiles) { //9yao
	return (GetMajorNum(tiles) >= 9);
}

bool TilePatternUtils::IsOrphan(const vector<int>& tiles) {
	return (GetMajorNum(tiles) >= 13);
}

S TilePatternUtils::GenScMsg (int index, const std::vector<int>& tens, const std::vector<int>& gains) {
	S scMsg = "sc=\"" + std::to_string(tens[index]) + "," + std::to_string(gains[index]);
	for (int j = 1; j < PlayerNum; j ++) {
		scMsg = scMsg + "," + std::to_string(tens[(index + j) %  PlayerNum]) + "," + std::to_string(gains[(index + j) % PlayerNum]);
	}
	scMsg += "\"";

	return scMsg;
}

std::vector<std::string> TilePatternUtils::GenOrphanMsg(int who, const std::vector<int>& tens, const std::vector<int>& tiles) {
	vector<S> rspMsgs;
	vector<int> gains(PlayerNum, 0);
	gains[who] = Mangan;

	S headMsg = "<AGARI ba=\"0, 0\"";
	S tailMsg = "/>";
	S haiMsg = "hai=\"" + std::to_string(tiles[0]);
	for (int i = 1; i < tiles.size(); i ++) {
		haiMsg = haiMsg + "," + std::to_string(tiles[i]);
	}
	haiMsg = haiMsg + "\"";
	S otherMsg = "ten=\"40,8000,1\" yaku=\"\" doraHai=\"0\" doraHaiUra=\"0\"";

	for (int i = 0; i < PlayerNum; i ++) {
		int whoIndex = GetMsgIndex(who, i);
		S whoMsg = "who=\"" + std::to_string(whoIndex) + "\" fromWho=\"" + std::to_string(whoIndex) + "\"";

		S scMsg = GenScMsg(i, tens, gains);

		rspMsgs.push_back(headMsg + " " + haiMsg + " " + otherMsg + " " + whoMsg + " " + scMsg + " " + tailMsg);
	}

	return rspMsgs;
}

vector<S> TilePatternUtils::Gen9YaoMsg(int index, std::vector<int>& tiles, std::vector<int>& tens) {
	vector<S> rspMsg;
	vector<int> gains(PlayerNum, 0);

	S headMsg = "<RYUUKYOKU type=\"yao9\" ba=\"0,0\"";
	for (int i = 0; i < PlayerNum; i ++) {
		S scMsg = GenScMsg(i, tens, gains);
		S haiHead = "hai" + std::to_string(GetMsgIndex(index, i));
		rspMsg.push_back(headMsg + " " + scMsg + " " + haiHead + " />");
	}

	return rspMsg;
}

std::string TilePatternUtils::GenProfMsg() {
	return "<PROF lobby=\"0\" type=\"1\" add=\"-19.0,0,0,1,0,0,6,2,1,2,1\"/>";
}
