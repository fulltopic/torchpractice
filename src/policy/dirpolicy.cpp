/*
 * dirpolicy.cpp
 *
 *  Created on: May 4, 2020
 *      Author: zf
 */


#include "policy/dirpolicy.h"

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include <torch/torch.h>

#include "tenhouclient/tenhoumsgparser.h"
#include "tenhouclient/tenhoustate.h"

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/path.hpp>
#include "../../include/utils/logger.h"

using namespace std;
using namespace torch;

using P = TenhouMsgParser;

DirPolicy::DirPolicy(string dirPath): path(dirPath),
		logger(Logger::GetLogger()),
		index(0),
		fileIndex(0)
{
}

DirPolicy::~DirPolicy() {}

vector<DirPolicyMsg> DirPolicy::parseTenhouFile(string fileName) {
	logger->info("To parse file {}", fileName);
	ifstream inputFile(fileName);
	string line;
	vector<DirPolicyMsg> msgs;

//	while (getline(inputFile, line)) {
//		if (line.find("INIT") != string::npos) {
//			break;
//		}
//	}

	while (getline(inputFile, line)) {
		if (line.find("Send") != string::npos) {
			if (line.find("<Z") != string::npos) {
				continue;
			}
			int sendPos = line.find("Send");
			int tokenPos = line.find(":", sendPos);
			string msg = line.substr(tokenPos + 2);
			msgs.push_back({0, msg});
			cout << "Send --> " << msg << endl;
		} else if (line.find("Get") != string::npos) {
			int getPos = line.find("Get");
			int tokenPos = line.find(":", getPos);
			string msg = line.substr(tokenPos + 2);
			if (msg.find("> <") != string::npos) {
				int tPos = msg.find("> <");
				string msg0 = msg.substr(0, tPos + 1);
				string msg1 = msg.substr(tPos + 2);
				msgs.push_back({1, msg0});
				msgs.push_back({1, msg1});
				cout << "Get --> " << msg0 << endl;
				cout << "Get --> " << msg1 << endl;
			} else {
				msgs.push_back({1, msg});
				cout << "Get --> " << msg << endl;
			}
		} else {
			cout << "Pass " << line << endl;
		}
	}

	return msgs;
}

void DirPolicy::init() {
	boost::filesystem::path dir(path);

	boost::filesystem::directory_iterator end_ite;
	for (boost::filesystem::directory_iterator ite(dir); ite != end_ite; ++ite) {
		if (boost::filesystem::is_regular_file (ite->status())) {
			string filePath = ite->path().string();
			logger->info("Get file {}", filePath);
			fileNames.push_back(filePath);
		}
	}
}

void DirPolicy::reset() {
	bool findNextFile = false;

	do {
		findNextFile = false;

		for (; index < msgs.size(); index ++) {
			logger->debug("Pass msg for init: {}", msgs[index].msg);
			if (msgs[index].msg.find("NEXTREADY") != string::npos) {
//			index ++;
				break;
			}
		}

		if ((index >= msgs.size())
				|| ((msgs.size() - index) < 10)) {
			logger->warn("To read next file ");
			string fileName = fileNames[fileIndex];
			fileIndex ++;

			msgs = parseTenhouFile(fileName);
			index = 0;
			findNextFile = true;


		}
	} while (findNextFile);




	for (; index < msgs.size(); index ++) {
		if (msgs[index].msg.find("NEXTREADY") == string::npos) {
			break;
		}
	}
	logger->debug("The next msg to sent: {}", msgs[index].msg);
}

int DirPolicy::getAction(Tensor values, vector<int> candidates) {
	static auto logger = Logger::GetLogger();

	for (; index < msgs.size(); index ++) {
		if (msgs[index].type == 0) {
			break;
		}
	}

	logger->debug("To get action from msg: {}", msgs[index].msg);
	string msg = msgs[index].msg;
	index ++;

//	GameMsgType msgType = P::GetMsgType(msg);
//	logger->debug("Get msg type: {}", msgType);
//
//	switch (msgType) {
	if (msg.find("<D") != string::npos)
	{
		logger->debug("To get drop tile");
//		auto rc = P::ParseDrop(msg);
//		int raw = rc.tile;
//		if (find(candidates.begin(), candidates.end(), P::Raw2Tile(raw)) == candidates.end()) {
//			cout << "Candidate failure " << endl;
//			return LstmStateAction::NOOPAction;
//		}
//		return P::Raw2Tile(raw);
		vector<string> dropMsgs = P::ParseItems(msg);
		string tileMsg = P::RemoveHead("p=\"", dropMsgs[1]);
		return P::Raw2Tile(stoi(tileMsg));
	} else if (msg.find("<N") != string::npos) {
		if (msg.find("<N />") != string::npos) {
			return LstmStateAction::NOOPAction;
		}

		auto nRc = P::ParseItems(msg);
		auto action = stoi(P::RemoveHead("type=\"", nRc[1]));
		cout << "Get action " << action << endl;
		if(action == 1) {
			return LstmStateAction::PongAction;
		} else if (action == 3) {
			return LstmStateAction::ChowAction;
		} else if (action == 2) {
			return LstmStateAction::KaKanAction;
		} else if (action == 4) { //Kan
			if (find(candidates.begin(), candidates.end(), MinKanAction) != candidates.end()) {
				logger->debug("Found Minkanaction ");
				return MinKanAction;
			} else {
				logger->error("does not found kan action in candidates");
				return LstmStateAction::NOOPAction;
			}
		} else if (action == 5) {
			if (find(candidates.begin(), candidates.end(), KaKanAction) != candidates.end()) {
					logger->debug("Found KaKanAction ");
					return KaKanAction;
			} else {
					logger->error("does not found kan action in candidates");
					return LstmStateAction::NOOPAction;
			}
		}
		else {
			cout << "Don't know corresponding action" << endl;
			return LstmStateAction::NOOPAction;
		}
	} else if (msg.find("REACH") != string::npos) {
		if (msg.find("hai") != string::npos) {
			return LstmStateAction::ReachAction;
		} else {
			logger->error("Unexpected message {}", msg);
			return LstmStateAction::NOOPAction;
		}
	} else {
		logger->error("Unexpected message: {}", msg);
		return LstmStateAction::NOOPAction;
	}
}


vector<int> DirPolicy::getTiles4Action(Tensor values, int actionType, vector<int> candidates, const int raw) {
	if (candidates.size() <= 2) {
		return candidates;
	}

	logger->debug("Get tiles for action {}", actionType);
	if (actionType == LstmStateAction::ChowAction) {
		int tmpIndex = index - 1;
		logger->debug("Get last msg {}, {}", msgs[tmpIndex].msg, msgs[tmpIndex].type);
		if (msgs[tmpIndex].type == 0) {
			string msg = msgs[tmpIndex].msg;
			logger->debug("Get tiles from msg {}", msg);
			vector<string> items = P::ParseItems(msg);
			vector<int> rcTiles;
			for (int i = 0; i < items.size(); i ++) {
				logger->debug("Item: {}", items[i]);
				if (items[i].find("hai0") != string::npos) {
					string haiMsg = P::RemoveHead("hai0=\"", items[i]);
					int hai0 = stoi(P::RemoveHead("hai0=\"", items[i]));
					logger->debug("Found rcTile {}", hai0);
					rcTiles.push_back(hai0);
				}
				if (items[i].find("hai1") != string::npos) {
					int hai1 = stoi(P::RemoveHead("hai1=\"", items[i]));
					logger->debug("Found rcTile {}", hai1);
					rcTiles.push_back(hai1);
				}
			}
			if (rcTiles.size() < 2) {
				logger->error("Failed to get 2 tiles ");
				return candidates;
			}

			if (find(candidates.begin(), candidates.end(), rcTiles[0]) != candidates.end()
					&& find(candidates.begin(), candidates.end(), rcTiles[1]) != candidates.end()) {
				logger->debug("Find candidates in expected msg ");
				return rcTiles;
			}
		}
	}

	logger->error("Something wrong in message and candidates ");
	return candidates;
}

int DirPolicy::getAction(torch::Tensor values, std::vector<int> candidates, std::vector<int> excludes) {
	return getAction(values, candidates);
}
