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
enum StorageIndex {
	InputIndex = 0,
	HStateIndex = 1,
	LabelIndex = 2,
	ActionIndex = 3,
	RewardIndex = 4,
};

class RlSetting {
public:
	static const int ProxyNum;
	static const int BatchSize;
	static const int UpdateThreshold;
	static const float ReturnGamma;

	static const int RnnSeqLen = 27;

	static const int NetNum;

	static std::vector<std::string> Names;

	static const std::string ServerIp;
	static const int ServerPort;
};
}



#endif /* INCLUDE_RLTEST_RLTESTSETTING_H_ */
