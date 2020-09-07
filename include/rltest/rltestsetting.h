/*
 * rltestsetting.h
 *
 *  Created on: Sep 7, 2020
 *      Author: zf
 */

#ifndef INCLUDE_RLTEST_RLTESTSETTING_H_
#define INCLUDE_RLTEST_RLTESTSETTING_H_

#include <vector>
#include <string>

namespace rltest {
class RlSetting {
	const int ProxyNum;
	const int BatchSize;
	const int UpdateThreshold;

	const int NetNum;

	const std::vector<const std::string> Names;
};
}



#endif /* INCLUDE_RLTEST_RLTESTSETTING_H_ */
