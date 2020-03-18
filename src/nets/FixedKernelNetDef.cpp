/*
 * FixedKernelNetDef.cpp
 *
 *  Created on: Feb 1, 2020
 *      Author: zf
 */


#include "nets/FixedKernelNetDef.h"
#include <cmath>

FixedKernelNetConf::FixedKernelNetConf(const int iLayerNum, const int initChan):
				layerNum(iLayerNum), initChannel(initChan) {
	std::cout << "In FixedKernelNetConf constructor " << std::endl;
	generateConfs();
}

int FixedKernelNetConf::calcConvKernelH(const int layer) {
	int lastOutputH = calcConvInputH(layer);
	auto kernelH = (lastOutputH >= DefaultConvKernelH? DefaultConvKernelH: lastOutputH);
	std::cout << "Lastoutput " << lastOutputH << ", " << kernelH << std::endl;

	return kernelH;
}

int FixedKernelNetConf::calcConvInputW(const int layer) {
	int lastLayerIndex = layer - 1;
	int lastInputW = convInputWs[lastLayerIndex];
	int lastKernelW = convKernelWs[lastLayerIndex];
//	int lastPad = convPadWs[lastLayerIndex];

	int lastOutputW = (lastInputW - lastKernelW + 1) / ConvPoolW;
	return lastOutputW;
}

int FixedKernelNetConf::calcConvInputH(const int layer) {
	int lastLayerIndex = layer - 1;
	int lastInputH = convInputHs[lastLayerIndex];
	int lastKernelH = convKernelHs[lastLayerIndex];
//	int lastPad = convPadHs[lastLayerIndex];

	int lastOutputH = (lastInputH - lastKernelH + 1) / DefaultConvPoolH;

	return lastOutputH;
}

void FixedKernelNetConf::fillInChanns() {
	convChanns.push_back(InputC); //Input layer
	convChanns.push_back(64); //conv layer 0
	convChanns.push_back(128);
	convChanns.push_back(192);
	convChanns.push_back(256);
	convChanns.push_back(512);
}

void FixedKernelNetConf::generateConfs() {
	fillInChanns();

	for (int i = 0; i <= layerNum; i ++) {
		convKernelWs.push_back(ConvKernelW);
//		convChanns.push_back(static_cast<int>(std::pow(2, i)) * initChannel);
//		if (i == 0) {
//			convChanns.push_back(InputC);
//		} else {
//			convChanns.push_back(static_cast<int>(std::pow(2, (i - 1))) * initChannel);
//		}
	}

	//Fill in 0th element
	if (InputH >= DefaultConvKernelH) {
		convKernelHs.push_back(DefaultConvKernelH);
	} else {
		convKernelHs.push_back(InputH);
	}
	convInputWs.push_back(InputW);
	convInputHs.push_back(InputH);
//	convPadWs.push_back(convKernelWs[0] / 2);
//	convPadHs.push_back(convKernelHs[0] / 2);

	//Fill in following layers
	for (int i = 1; i <= layerNum; i ++) {
		convKernelHs.push_back(calcConvKernelH(i));
		convInputWs.push_back(calcConvInputW(i));
		convInputHs.push_back(calcConvInputH(i));
//		convPadWs.push_back(convKernelWs[i] / 2);
//		convPadHs.push_back(convKernelHs[i] / 2);
	}


	std::cout << "FixedKernelNetConf " << convInputHs.size() << std::endl;

	for (int i = 0; i <= layerNum; i ++) {
		std::cout << "" << i << ", " << convChanns[i] << ", " << convInputWs[i]
			<< ", " << convInputHs[i] << " = "
			<< convChanns[i] * convInputWs[i] * convInputHs[i]<< std::endl;
	}
}

int FixedKernelNetConf::getFcInput() {
	int layerIndex = layerNum;
//	std::cout << "FcInput: " << convChanns[layerIndex] * convInputWs[layerIndex] * convInputHs[layerIndex] << std::endl;
//	std::cout << "" << layerIndex << ", " << convChanns[layerIndex] << ", " << convInputWs[layerIndex] << ", " << convInputHs[layerIndex] << std::endl;
//	for (int i = 0; i <= layerIndex; i ++) {
//		std::cout << convInputWs[i] << ", " << std::endl;
//	}
	return convChanns[layerIndex] * convInputWs[layerIndex] * convInputHs[layerIndex];
}
