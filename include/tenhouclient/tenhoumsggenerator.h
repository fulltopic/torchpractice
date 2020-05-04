/*
 * tenhoumsggenerator.h
 *
 *  Created on: Apr 11, 2020
 *      Author: zf
 */

#ifndef INCLUDE_TENHOUCLIENT_TENHOUMSGGENERATOR_H_
#define INCLUDE_TENHOUCLIENT_TENHOUMSGGENERATOR_H_

#include "tenhouclient/tenhouconsts.h"

#include <boost/algorithm/string.hpp>
#include <iostream>
#include <string>
#include <vector>

class TenhouMsgGenerator {
private:
	TenhouMsgGenerator() = delete;
	TenhouMsgGenerator(TenhouMsgGenerator&) = delete;
	TenhouMsgGenerator& operator=(TenhouMsgGenerator&) = delete;
	~TenhouMsgGenerator() = delete;

public:
	static int64_t GetTransValue(int index);

	static std::string AddWrap(std::string msg);
	static std::string AddHead(std::string head, int tile);
	static std::string AddHead(std::string head, std::string msg);

	static std::string GenConnMsg();

	static std::string GenHeloMsg(std::string userName);
	static std::string GenAuthReply(std::vector<std::string> parts);
	static std::string GenPxrMsg();
	static std::string GenJoinMsg();
	static std::string GenRejoinMsg(std::string msg);
	static std::string GenGoMsg();
	static std::string GenNextReadyMsg();

	static std::string GenKAMsg();

	static std::string GenByeMsg();

	static std::string GenDropMsg(const int tile);
	static std::string GenReachMsg(int tile);
	static std::string GenRonMsg(int indType);
	static std::string GenNoopMsg();
	static std::string GenStealMsg(int type, std::vector<int> tiles);
	static std::string GenPongMsg(std::vector<int> tiles);
	static std::string GenChowMsg(std::vector<int> tiles);
	static std::string GenKanMsg(int raw);
};


#endif /* INCLUDE_TENHOUCLIENT_TENHOUMSGGENERATOR_H_ */
