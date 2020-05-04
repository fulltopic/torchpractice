/*
 * tenhoumsgutils.cpp
 *
 *  Created on: Apr 14, 2020
 *      Author: zf
 */

#include "tenhouclient/tenhoumsgutils.h"

using namespace std;

bool TenhouMsgUtils::IsTerminalMsg(const string msg) {
	return (msg.find("AGARI") != string::npos)
			|| (msg.find("RYU") != string::npos);
}

bool TenhouMsgUtils::IsGameMsg(const string msg) {
	return true;
}


