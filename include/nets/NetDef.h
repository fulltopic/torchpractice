/*
 * NetDef.h
 *
 *  Created on: Jan 7, 2020
 *      Author: zf
 */

#ifndef INCLUDE_NETDEF_H_
#define INCLUDE_NETDEF_H_

#include <iostream>
#include <string>
#include <tuple>

enum MjNetConf {
	InputW = 72,
	InputH = 5,
	InputC = 1,

	Conv0InChan = InputC,
	Conv0OutChan = 64,
	Conv0KernelW = 3,
	Conv0KernelH = 5,
	Conv0StrideW = 1,
	Conv0StrideH = 1,
	Conv0PoolW = 2,
	Conv0PoolH = 1,
	Conv0PoolPadW = 0,
	Conv0PoolPadH = 0,
	Conv0PoolSize = 2,

	Conv1InChan = Conv0OutChan,
	Conv1OutChan = 64,
	Conv1KernelW = 3,
	Conv1KernelH = 1,
	Conv1StrideW = 1,
	Conv1StrideH = 1,
	Conv1PoolPadW = 0,
	Conv1PoolPadH = 0,
	Conv1PoolSize = 1,

	Lstm0Hidden = 100,
	Lstm0Layer = 1,
	Lstm0Dir = 1,

	FcInput = Lstm0Hidden,
	FcOutput = 42,

	OutputMax = 1,
};


const std::pair<int, int> GetConvOutput(const int convInputW, const int convInputH,
					const int convKernelW, const int convKernelH, const int poolW, const int poolH, const int poolPadH,
					const int convStrideW, const int convStrideH);
//							{
//	int convOutputW = (convInputW - convKernelW + 1) / convStrideW;
//	int convOutputH = (convInputH - convKernelH + 1 + poolPadH * 2) / convStrideH;
//	int convPoolOutputW = convOutputW / poolW;
//	int convPoolOutputH = convOutputH / poolH;
//
//	return std::make_pair(convPoolOutputW, convPoolOutputH);
//}

 const int GetLstm0Input() ;
//{
//	int convOutputW = 0;
//	int convOutputH = 0;
//	std::tie(convOutputW, convOutputH) = GetConvOutput(InputW, InputH,
//			Conv0KernelW, Conv0KernelH, Conv0PoolW, Conv0PoolH, Conv0PoolPadH, Conv0StrideW, Conv0StrideH);
//
//	std::tie(convOutputW, convOutputH) = GetConvOutput(convOutputW, convOutputH,
//			Conv1KernelW, Conv1KernelH, Conv1PoolSize, Conv1PoolSize, Conv1PoolPadH, Conv1StrideW, Conv1StrideH);
//
//	return convOutputW * convOutputH * Conv1OutChan;
//}



#endif /* INCLUDE_NETDEF_H_ */
