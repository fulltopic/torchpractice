/*
 * NetDef.cpp
 *
 *  Created on: Jan 22, 2020
 *      Author: zf
 */


#include "nets/NetDef.h"

const std::pair<int, int> GetConvOutput(const int convInputW, const int convInputH,
					const int convKernelW, const int convKernelH, const int poolW, const int poolH, const int poolPadH,
					const int convStrideW, const int convStrideH) {
	int convOutputW = (convInputW - convKernelW + 1) / convStrideW;
	int convOutputH = (convInputH - convKernelH + 1 + poolPadH * 2) / convStrideH;
	int convPoolOutputW = convOutputW / poolW;
	int convPoolOutputH = convOutputH / poolH;

	return std::make_pair(convPoolOutputW, convPoolOutputH);
}

const int GetLstm0Input() {
	int convOutputW = 0;
	int convOutputH = 0;
	std::tie(convOutputW, convOutputH) = GetConvOutput(InputW, InputH,
			Conv0KernelW, Conv0KernelH, Conv0PoolW, Conv0PoolH, Conv0PoolPadH, Conv0StrideW, Conv0StrideH);

	std::tie(convOutputW, convOutputH) = GetConvOutput(convOutputW, convOutputH,
			Conv1KernelW, Conv1KernelH, Conv1PoolSize, Conv1PoolSize, Conv1PoolPadH, Conv1StrideW, Conv1StrideH);

	return convOutputW * convOutputH * Conv1OutChan;
}
