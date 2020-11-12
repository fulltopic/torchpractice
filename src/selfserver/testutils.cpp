/*
 * testutils.cpp
 *
 *  Created on: Nov 5, 2020
 *      Author: zf
 */

#define private public

#include "selfserver/agarichecker.h"
#include "selfserver/playerstate.h"

#include "utils/logger.h"
#include "spdlog/spdlog.h"
#include "tenhouclient/tenhoumsgparser.h"


#include <iostream>
#include <string>

namespace {
void testAgariChecker() {
//	std::vector<int> tiles {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5};
	std::vector<int> tiles {0, 3, 3, 3, 3, 2, 0, 0, 0, 0,
							0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
							0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
							0, 0, 0, 0
	};


	auto rc = AgariChecker::GetInst()->getKey(tiles);
	int key = std::get<0>(rc);
	std::vector<int> combs = std::get<1>(rc);

	bool isAgari = AgariChecker::GetInst()->isAgari(key);

	std::cout << "IsAgari " << isAgari << std::endl;
}

void stateAgari() {
	PlayerState state(0, 0);
//	state.totalTiles = {0, 2, 3, 3, 3, 2, 0, 0, 0, 0,
//			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//			0, 0, 0, 0
//	};
//	bool isAgari = state.checkAgari(5);
//	std::cout << isAgari << std::endl;
//
//	state.totalTiles = {0, 2, 4, 4, 3, 2, 0, 0, 0, 0,
//			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//			0, 0, 0, 0
//	};
//	isAgari = state.checkAgari(5);
//	std::cout << isAgari << std::endl;
//
//	state.totalTiles = {0, 3, 4, 3, 3, 1, 0, 0, 0, 0,
//			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//			0, 0, 0, 0
//	};
//	isAgari = state.checkAgari(22);
//	std::cout << isAgari << std::endl;
//
//	state.totalTiles = {0, 2, 2, 2, 2, 2, 2, 1, 0, 0,
//			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//			0, 0, 0, 0
//	};
//	isAgari = state.checkAgari(30);
//	std::cout << isAgari << std::endl;
//
//	state.totalTiles = {0, 4, 4, 4, 4, 1, 0, 0, 0, 0,
//			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//			0, 0, 0, 0
//	};
//	isAgari = state.checkAgari(22);
//	std::cout << isAgari << std::endl;

	state.totalTiles = {0, 1, 1, 1, 0, 1, 0, 0, 0, 0,
						3, 0, 1, 1, 0, 0, 0, 0, 0, 0,
						0, 1, 1, 1, 1, 0, 0, 0, 0, 0,
						1, 0, 0, 0};
	bool isAgari = state.checkAgari(42);
	std::cout << isAgari << std::endl;
}

//state.closeTiles = {0, 0, 0, 0, 0, 0, 0, 0, 0,
//					0, 0, 0, 0, 0, 0, 0, 0, 0,
//					0, 0, 0, 0, 0, 0, 0, 0, 0,
//					0, 0, 0, 0, 0, 0, 0
//};
void testReach() {
	PlayerState state(0, 0);
	state.closeTiles = {0, 0, 0, 0, 0, 0, 0, 0, 0,
						0, 0, 0, 0, 0, 0, 0, 0, 0,
						0, 0, 0, 0, 0, 0, 0, 0, 0,
						0, 0, 0, 0, 0, 0, 0
	};
	bool isReach = state.checkReach();
	std::cout << isReach << std::endl; //0

	state.closeTiles = {0, 0, 0, 1, 1, 1, 0, 0, 0,
						0, 2, 0, 0, 0, 1, 2, 0, 0,
						0, 1, 1, 1, 1, 1, 1, 0, 0,
						0, 0, 0, 0, 0, 0, 0
	};
	isReach = state.checkReach();
	std::cout << isReach << std::endl; //1

	state.closeTiles = {0, 1, 1, 1, 0, 0, 0, 0, 0,
							0, 0, 0, 0, 0, 0, 0, 0, 2,
							0, 0, 1, 1, 1, 1, 1, 1, 0,
							0, 0, 0, 0, 0, 0, 2
					};
	isReach = state.checkReach();
	std::cout << isReach << std::endl; //0, tileNum = 13

	state.closeTiles = {0, 0, 0, 2, 2, 1, 2, 1, 0,
						0, 0, 0, 0, 0, 0, 0, 0, 0,
						0, 0, 1, 0, 1, 1, 1, 1, 0,
						0, 0, 0, 0, 0, 0, 0
	};
	isReach = state.checkReach();
	std::cout << isReach << std::endl;//0

	state.closeTiles = {0, 0, 0, 0, 2, 0, 0, 0, 0,
						0, 0, 0, 0, 0, 0, 0, 0, 0,
						0, 0, 0, 3, 0, 0, 0, 0, 0,
						0, 0, 3, 2, 0, 1, 3
	};
	isReach = state.checkReach();
	std::cout << isReach << std::endl; //1

	state.closeTiles = {0, 0, 0, 3, 1, 2, 0, 1, 0,
						0, 0, 3, 0, 0, 1, 0, 3, 0,
						0, 0, 0, 0, 0, 0, 0, 0, 0,
						0, 0, 0, 0, 0, 0, 0
	};
	isReach = state.checkReach();
	std::cout << isReach << std::endl; //1

	state.closeTiles = {0, 1, 1, 1, 2, 2, 3, 0, 0,
						0, 0, 0, 0, 0, 0, 0, 0, 0,
						0, 0, 0, 2, 0, 0, 0, 2, 0,
						0, 0, 0, 0, 0, 0, 0
	};
	isReach = state.checkReach();
	std::cout << isReach << std::endl; //1

	state.closeTiles = {0, 1, 1, 1, 1, 1, 1, 0, 0,
							0, 1, 1, 1, 0, 1, 0, 0, 0,
							0, 0, 0, 0, 0, 1, 1, 1, 0,
							1, 0, 0, 0, 0, 0, 0
	};
	isReach = state.checkReach();
	std::cout << isReach << std::endl; //1


	state.closeTiles = {1, 0, 1, 1, 2, 1, 0, 0, 0,
						0, 1, 1, 0, 1, 1, 0, 0, 0,
						0, 0, 0, 0, 0, 2, 0, 0, 0,
						0, 0, 1, 0, 0, 0, 0
	};
	isReach = state.checkReach();
	std::cout << isReach << std::endl; //0

	state.closeTiles = {0, 0, 0, 0, 2, 0, 2, 0, 2,
						0, 0, 0, 1, 1, 1, 0, 0, 0,
						0, 0, 0, 1, 1, 1, 0, 0, 2,
						0, 0, 0, 0, 0, 0, 0};
	isReach = state.checkReach();
	std::cout << isReach << std::endl; //0

	state.closeTiles = {0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
						3, 0, 0, 0, 4, 0, 0, 0, 0, 0,
						0, 1, 0, 3, 0, 0, 0, 0, 0, 2,
						0, 0, 0, 0, };
	isReach = state.checkReach();
	std::cout << isReach << std::endl; //0

}

//TODO: Check M
void testM() {
	int type = 1;
	int raw = 65;
	std::vector<int> raws{66, 67};
	int m = PlayerState::GetM(type, raw, raws);
	std::cout << "m = " << m << std::endl << std::endl;

	type = 3;
	raw = 13;
	raws = {17, 21};
	m = PlayerState::GetM(type, raw, raws);
	std::cout << "m = " << m << std::endl << std::endl;;

	type = 1;
	raw = 132;
	raws = {133, 134};
	m = PlayerState::GetM(type, raw, raws);
	std::cout << "m = " << m << std::endl << std::endl;

	type = 3;
	raw = 61;
	raws = {53, 59};
	m = PlayerState::GetM(type, raw, raws);
	std::cout << "m = " << m << std::endl << std::endl;

	type = 3;
	raw = 77;
	raws = {75, 83};
	m = PlayerState::GetM(type, raw, raws);
	std::cout << "m = " << m << std::endl << std::endl;

	type = 3;
	raw = 30;
	raws = {22, 27};
	m = PlayerState::GetM(type, raw, raws);
	std::cout << "m = " << m << std::endl << std::endl;

	type = 1;
	raw = 85;
	raws = {86, 87};
	m = PlayerState::GetM(type, raw, raws);
	std::cout << "m = " << m << std::endl << std::endl;
}

void testParse() {
	std::string msg = "<N type=\"1\" hai0=\"120\" hai1=\"122\" />";
	auto items = TenhouMsgParser::ParseItems(msg);
	std::cout << "items size = " << items.size() << std::endl;

	for (int i = 0; i < items.size(); i ++) {
		std::cout << "To parse " << items[i] << std::endl;
		if (items[i].find("hai0") != std::string::npos) {
			int hai = TenhouMsgParser::ParseHead("hai0=\"", items[i]);
			std::string head = "hai0=\"";
			std::string valueMsg = items[i].substr(head.size(), msg.size() - head.size());
			std::cout << "value Msg " << valueMsg << std::endl;
			valueMsg = valueMsg.substr(0, valueMsg.size() - 1);
			std::cout << "value Msg " << valueMsg << std::endl;
			std::cout << atoi(valueMsg.c_str()) << std::endl;
			std::cout << "Get hai " << hai << std::endl;
		}
	}
}
}

int main(int argc, char** argv) {
	spdlog::set_level(spdlog::level::info);

//	testAgariChecker();
//	stateAgari();
	testReach();
//	testM();
//	testParse();
}

