/*
 * tenhoumsgutils.h
 *
 *  Created on: Apr 14, 2020
 *      Author: zf
 */

#ifndef INCLUDE_TENHOUCLIENT_TENHOUMSGUTILS_H_
#define INCLUDE_TENHOUCLIENT_TENHOUMSGUTILS_H_

#include <string>
#include <vector>
#include <set>
#include <map>

class TenhouMsgUtils {
public:
	static bool IsTerminalMsg (const std::string);
	static bool IsGameMsg (const std::string);
};



#endif /* INCLUDE_TENHOUCLIENT_TENHOUMSGUTILS_H_ */
