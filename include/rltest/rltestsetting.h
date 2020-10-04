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
public:
	static const int ProxyNum;
	static const int BatchSize;
	static const int UpdateThreshold;
	static const float ReturnGamma;

//	static const int RnnSeqLen = 27;

	static const int SaveEpochThreshold;
	static const int ThreadNum;

//	static const int NetNum;

	static std::vector<std::string> Names;
	static const std::string ModelDir;
	static const std::string StatsDataName;
	static const std::string ServerIp;
	static const int ServerPort;

	static const bool IsPrivateTest;
};
}



#endif /* INCLUDE_RLTEST_RLTESTSETTING_H_ */
