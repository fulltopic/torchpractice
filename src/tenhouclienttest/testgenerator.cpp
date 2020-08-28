/*
 * testgenerator.cpp
 *
 *  Created on: Apr 12, 2020
 *      Author: zf
 */



#include "tenhouclient/tenhoumsggenerator.h"

#include <vector>
#include <set>
#include <map>

#include <string>
#include <iostream>
#include <sstream>
#include <algorithm>

using namespace std;

using G = TenhouMsgGenerator;

#define Match {testMatch(msg, eMsg);}

void testAdd() {
	string a = "a ";
	string msg = "HELO " + a;
	cout << msg << endl;
}

void testMatch(const string msg, const string eMsg) {
	cout << msg << endl;
	cout << eMsg << endl;
	cout << msg.compare(eMsg) << endl;
}

void testHelo() {
	string expectedMsg = "<HELO name=\"NoName\" tid=\"f0\" sx=\"M\" />";
	string msg = TenhouMsgGenerator::GenHeloMsg("NoName");
	cout << msg << endl;
	cout << msg.compare(expectedMsg) << endl;
}

void testAuthReply() {
	vector<string> parts {
		"20200412",
		"09217af6"
	};

	string msg = TenhouMsgGenerator::GenAuthReply(parts);
	string expectedMsg = "<AUTH val=\"20200412-2f7f20a7\"/>";

	testMatch(msg, expectedMsg);
}

void testPxrMsg() {
	string msg = TenhouMsgGenerator::GenPxrMsg();
	string eMsg = "<PXR V=\"1\" />";

	testMatch(msg, eMsg);
}

void testJoinMsg() {
	string msg = TenhouMsgGenerator::GenJoinMsg();
	string eMsg = "<JOIN t=\"0,1\" />";

	Match
}

void testRejoinMsg() {
	string msg = TenhouMsgGenerator::GenRejoinMsg("<REJOIN t=\"0,1,r\"/>");
	cout << "Get rejoin msg " << msg << endl;

	string eMsg = "<JOIN t=\"0,1,r\" />";

	cout << msg.length() << " : " << eMsg.length() << endl;
	for (int i = 0; i < min(msg.length(), eMsg.length()); i ++) {
		cout << msg.c_str()[i] << " : " << eMsg.c_str()[i] << endl;
	}

	if (msg.length() > eMsg.length()) {
		cout << msg.at(msg.length() - 1) << endl;
	} else {
		cout << eMsg.at(eMsg.length() - 1) << endl;
	}
}
void testGoMsg() {
	auto msg = G::GenGoMsg();
	auto eMsg = "<GOK />";

	Match
}

void testNextMsg() {
	auto msg = G::GenNextReadyMsg();
	auto eMsg = "<NEXTREADY />";

	Match
}

void testByeMsg() {
	auto msg = G::GenByeMsg();
	auto eMsg = "<BYE />";

	Match
}

void testKaMsg() {
	auto msg = G::GenKAMsg();
	auto eMsg = "<Z />";

	Match
}

void testNoopMsg() {
	auto msg = G::GenNoopMsg();
	auto eMsg = "<N />";

	Match
}

void testDropMsg() {
	auto msg = G::GenDropMsg(33);
	auto eMsg = "<D p=\"33\"/>";

	Match
}

void testPongMsg() {
	auto msg = G::GenPongMsg({129, 131});
	auto eMsg = "<N type=\"1\" hai0=\"129\" hai1=\"131\" />";

	Match
}

void testChowMsg() {
	auto msg = G::GenChowMsg({61, 64});
	auto eMsg = "<N type=\"3\" hai0=\"61\" hai1=\"64\" />";

	Match
}

void testReachMsg() {
	auto msg = G::GenReachMsg(16);
	auto eMsg = "<REACH hai=\"16\" />";

	Match
}

void testRonMsg() {
	auto msg = G::GenRonMsg(6);
	auto eMsg = "<N type=\"6\" />";

	Match
}

int main() {
//	testHelo();
//	testAuthReply();
//	testPxrMsg();
//	testJoinMsg();
//	testGoMsg();
//	testNextMsg();
//	testByeMsg();
//	testKaMsg();
//	testNoopMsg();
//	testDropMsg();
//	testPongMsg();
//	testChowMsg();
//	testReachMsg();
//	testRonMsg();
	testRejoinMsg();
}
