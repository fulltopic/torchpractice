/*
 * tenhoumsggenerator.cpp
 *
 *  Created on: Apr 11, 2020
 *      Author: zf
 */

#include "tenhouclient/tenhoumsggenerator.h"

#include <vector>
#include <set>
#include <map>

#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>

#include <boost/algorithm/string.hpp>


using namespace std;
using namespace boost;

int64_t TenhouMsgGenerator::GetTransValue(int index) {
	static vector<int64_t> translationTable = {63006, 9570, 49216, 45888, 9822, 23121, 59830, 51114, 54831, 4189, 580, 5203, 42174, 59972,
		    55457, 59009, 59347, 64456, 8673, 52710, 49975, 2006, 62677, 3463, 17754, 5357};

	return translationTable[index];
}

std::string TenhouMsgGenerator::AddWrap(std::string msg) {
	return "<" + msg + "/>\0";
}

string TenhouMsgGenerator::AddHead(string head, int tile) {
	return head + "=\"" + to_string(tile) + "\"";
}

string TenhouMsgGenerator::AddHead(string head, string msg) {
	return head + "=\"" + msg + "\"";
}

string TenhouMsgGenerator::GenHeloMsg(string userName) {
	string msg = "HELO name=\"" + userName + "\" tid=\"f0\" sx=\"M\" ";
	return AddWrap(msg);
}

string TenhouMsgGenerator::GenAuthReply(vector<string> parts) {
	int index = stoi("2" + parts[0].substr(2))
			% (12 - stoi(parts[0].substr(parts[0].size() - 1, 1)))
			* 2;
	int a = GetTransValue(index) ^ stoi(parts[1].substr(0, 4), 0, 16);
	int b = GetTransValue(index + 1) ^ stoi(parts[1].substr(4, 4), 0, 16);

	ostringstream postfixStream;
	postfixStream << hex << setw(2) << a << hex << setw(2) << b;
	string postfix = postfixStream.str();
	to_lower(postfix);

	return AddWrap("AUTH " + AddHead("val", parts[0] + "-" + postfix));
}

string TenhouMsgGenerator::GenPxrMsg() {
	return AddWrap("PXR V=\"1\" ");
}

string TenhouMsgGenerator::GenJoinMsg() {
	return AddWrap("JOIN t=\"0,1\" ");
}

string TenhouMsgGenerator::GenRejoinMsg(string msg) {
//	trim(msg);
//	replace_first(msg, "REJOIN", "JOIN");
//	replace_first(msg, "/>", " />");
//	return msg;

	return AddWrap("JOIN t=\"0,1,r\" ");
}

std::string TenhouMsgGenerator::GenLobbyPxrMsg() {
	return AddWrap("PXR v=\"-1\"");
}

std::string TenhouMsgGenerator::GenLobbyJoinMsg() {
	return AddWrap("JOIN t=\"3581,1\"");
}

std::string TenhouMsgGenerator::GenLobbyRejoinMsg() {
	return AddWrap("JOIN t=\"3581,1\"");
}

std::string TenhouMsgGenerator::GenLobbyChatMsg() {
	return AddWrap("CHAT text=\"/lobby 3581\"");
}

string TenhouMsgGenerator::GenGoMsg() {
	return AddWrap("GOK ");
}


string TenhouMsgGenerator::GenNextReadyMsg() {
	return AddWrap("NEXTREADY ");
}

string TenhouMsgGenerator::GenByeMsg() {
	return AddWrap("BYE ");
}

string TenhouMsgGenerator::GenKAMsg() {
	return AddWrap("Z ");
}

string TenhouMsgGenerator::GenNoopMsg() {
	return AddWrap("N ");
}

string TenhouMsgGenerator::GenDropMsg(int tile) {
	return AddWrap("D " + AddHead("p", tile));
}

string TenhouMsgGenerator::GenStealMsg(int type, vector<int> tiles) {
	string typeMsg = AddHead("type", type);
	string haiMsg("");
	for (int i = 0; (i < tiles.size()) && (i < 2); i ++) {
		haiMsg += AddHead("hai" + to_string(i), tiles[i]);
		haiMsg += " ";
	}

	return AddWrap("N " + typeMsg + " " + haiMsg);
}

string TenhouMsgGenerator::GenPongMsg(vector<int> tiles) {
	return GenStealMsg(1, tiles);
}

string TenhouMsgGenerator::GenChowMsg(vector<int> tiles) {
	return GenStealMsg(3, tiles);
}

string TenhouMsgGenerator::GenKanMsg(int raw) {
	return AddWrap("N type=\"2\" ");
}

string TenhouMsgGenerator::GenReachMsg(int tile) {
	return AddWrap("REACH " + AddHead("hai", tile) + " ");
}

string TenhouMsgGenerator::GenRonMsg(int indType) {
	int type = 6;
	switch(indType) {
	//win by self
	case 16:
	case 48:
		type = 7;
		break;
	case 64: //yao9
		type = 9;
		break;
	default:
		//win by other drop
		type = 6;
		break;
	}

	return AddWrap("N " + AddHead("type", type) + " ");
}
