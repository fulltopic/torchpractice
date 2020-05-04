/*
 * testparser.cpp
 *
 *  Created on: Apr 10, 2020
 *      Author: zf
 */

#include "tenhouclient/tenhoumsgparser.h"
#include "tenhouclient/tenhouconsts.h"
#include <iostream>
#include <boost/xpressive/xpressive.hpp>


using namespace boost::xpressive;


using namespace std;

void testInit() {
	string msg = "<INIT seed=\"0,0,0,1,3,86\" ten=\"250,250,250,250\" oya=\"0\" hai0=\"121,73,114,48,129,53,126,102,46,33,85,123,79\" hai1=\"124,84,61,89,51,81,108,110,98,104,113,23,57\" hai2=\"100,39,35,7,58,96,130,27,30,6,17,69,22\" hai3=\"52,111,78,122,92,125,60,28,47,103,18,106,36\" shuffle=\"mt19937ar,4324C796,531733FC,62506C1A,2FDCCC72,316E5D1A,FD7D2F48,8857571D,1FD22674\"/>";
	InitResult rc = TenhouMsgParser::ParseInit(msg);
	cout << "owner: " << rc.oyaIndex << endl;
	for (int i = 0; i < rc.tiles.size(); i ++) {
		cout << rc.tiles[i] << ", ";
	}
	cout << endl;
}

void testDrop() {
	string msg = "<E51/>";
	DropResult rc = TenhouMsgParser::ParseDrop(msg);

	cout << "player " << rc.playerIndex << ", dropped " << rc.tile << endl;
}

void testAccept() {
	string msg = "<T40/>";
	AcceptResult rc = TenhouMsgParser::ParseAccept(msg);

	cout << "Accept " << rc << endl;
}

void testSteal() {
//	string msg = "<N who=\"1\" m=\"42538\"/>"; // pong, wrong
//	string msg = "<N who=\"1\" m=\"62551\"/>"; //Chow
//	string msg = "<N who=\"1\" m=\"42546\"/>"; //kakan
//	string msg = "<N who=\"2\" m=\"12298\"/>"; //pong, OK
	string msg = "<N who=\"1\" m=\"63527\"/>"; //Chow
	StealResult rc = TenhouMsgParser::ParseSteal(msg);
	string stealType = "Invalid";
	switch(rc.flag) {
	case ChowFlag:
		stealType = "Chow";
		break;
	case PongFlag:
		stealType = "Pong";
		break;
	case KakanFlag:
		stealType = "Kakan";
		break;
	case AnkanFlag:
		stealType = "Ankan";
		break;
	case KitaBits:
		stealType = "Kita";
		break;
	}

	cout << stealType << ": " << rc.playerIndex << ", " << rc.stealTile << endl;
	for (int i = 0; i < rc.tiles.size(); i ++) {
		cout << rc.tiles[i] << ", ";
	}
	cout << endl;
}

void testReach() {
//	string msg = "<REACH who=\"2\" step=\"1\"/>";
	string msg = "<REACH who=\"1\" step=\"2\"/>";
	ReachResult rc = TenhouMsgParser::ParseReach(msg);

	cout << rc.playerIndex << ", " << rc.reachPhase << endl;
}

void testAgari() {
	string msg = "<AGARI ba=\"0,1\" hai=\"23,25,30,41,46,51,59,60,64,66,67,81,86,91\" machi=\"67\" ten=\"30,3900,0\" yaku=\"1,1,7,1,8,1,53,0\" doraHai=\"33,118\" doraHaiUra=\"117,50\" who=\"1\" fromWho=\"3\" sc=\"230,0,123,49,330,0,307,-39\"/>";
	auto rc = TenhouMsgParser::ParseAgari(msg);

	cout << rc.winnerIndex << ", " << rc.reward << endl;
}

void testRyu() {
	string msg = "<RYUUKYOKU ba=\"1,1\" sc=\"191,30,172,-10,330,-10,297,-10\" hai0=\"4,8,13,40,44,48,54,55,81,83\"/>";
	auto rc = TenhouMsgParser::ParseRyu(msg);

	cout << rc << endl;
}

void testReg() {
//	sregex rx = sregex::compile("</u|v");
	sregex rx = as_xpr('<') >> icase(as_xpr('u') | 'v' | 'd') >> +_d;
	smatch what;
	string str ("<u32/>");
	string str1 ("<v49/>");
	string str2 ("u34");
	string str3 ("<vlij/>");
	string str4 ("<D89/>");
	string str5 ("<D 89/>");
//	auto rc = regex_search(str, what, rx);
//	cout << str << ": " << rc << endl;
//	rc = regex_search(str1, what, rx);
//	cout << str1 << ": " << rc << endl;
//	rc = regex_search(str2, what, rx);
//	cout << str2 << ": " << rc << endl;
//	rc = regex_search(str3, what, rx);
//	cout << str3 << ": " << rc << endl;
//	rc = regex_search(str4, what, rx);
//	cout << str4 << ": " << rc << endl;

	auto pr = [&](string s) {
		auto rc = regex_search(s, what, rx);
		cout << s << ": " << rc << endl;
	};

	pr(str);
	pr(str1);
	pr(str2);
	pr(str3);
	pr(str4);
	pr(str5);

	sregex rx1 = as_xpr(' ') >> "t=\"" >> +_d;
	string str6 ("<G28 t=\"1\"/>");
	cout << str6 << ": " << regex_search(str6, what, rx1) << endl;
}

int main() {
//	testInit();
//	testDrop();
//	testAccept();
//	testSteal();
//	testReach();
//	testAgari();
//	testRyu();
	testReg();
}


