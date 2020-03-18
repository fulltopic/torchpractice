/*
 * FixedKernelNetDef.h
 *
 *  Created on: Feb 1, 2020
 *      Author: zf
 */

#ifndef INCLUDE_NETS_FIXEDKERNELNETDEF_H_
#define INCLUDE_NETS_FIXEDKERNELNETDEF_H_

#include <iostream>
#include <string>
#include <vector>
#include <tuple>

enum FixedKernelNetConfEnum {
	InputW = 72,
	InputH = 5,
	InputC = 1,

	ConvKernelW = 3,
	ConvPoolW = 1,
	DefaultConvKernelH = 3,
	DefaultConvPoolH = 1,

	Conv0InChan = InputC,

	InitChan = 64,
	LayerNum = 4,
//	Conv0OutChan = 64,
//	Conv0KernelW = 3,
//	Conv0KernelH = 5,
//	Conv0StrideW = 1,
//	Conv0StrideH = 1,
//	Conv0PoolW = 2,
//	Conv0PoolH = 1,
//	Conv0PoolPadW = 0,
//	Conv0PoolPadH = 0,
//	Conv0PoolSize = 2,
//
//	Conv1InChan = Conv0OutChan,
//	Conv1OutChan = 64,
//	Conv1KernelW = 3,
//	Conv1KernelH = 1,
//	Conv1StrideW = 1,
//	Conv1StrideH = 1,
//	Conv1PoolPadW = 0,
//	Conv1PoolPadH = 0,
//	Conv1PoolSize = 1,
//
//	Lstm0Hidden = 100,
//	Lstm0Layer = 1,
//	Lstm0Dir = 1,
//
//	FcInput = Lstm0Hidden,
	FcOutput = 42,

	OutputMax = 1,
};

class FixedKernelNetConf {
public:
	const int layerNum; //[1, N]
	const int initChannel;
	std::vector<int> convKernelWs;
	std::vector<int> convKernelHs;
//	std::vector<int> convPadWs;
//	std::vector<int> convPadHs;
	std::vector<int> convChanns;
	std::vector<int> convInputWs;
	std::vector<int> convInputHs;

	FixedKernelNetConf(const int iLayerNum, const int initChan);
	~FixedKernelNetConf() = default;
	int getFcInput();

private:
	int calcConvKernelH(const int layer);
	int calcConvInputW(const int layer);
	int calcConvInputH(const int layer);
	void fillInChanns();
	void generateConfs();
};



#endif /* INCLUDE_NETS_FIXEDKERNELNETDEF_H_ */
