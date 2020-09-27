/*
 * rltestsetting.cpp
 *
 *  Created on: Sep 7, 2020
 *      Author: zf
 */


#include "rltest/rltestsetting.h"

namespace rltest {

const int RlSetting::ProxyNum = 3;
const int RlSetting::BatchSize = 8;
const int RlSetting::UpdateThreshold = 4; //TODO: dirty data

const float RlSetting::ReturnGamma = 0.9;

const int RlSetting::NetNum = 2;

const std::string RlSetting::ServerIp = "133.242.10.78";
const int RlSetting::ServerPort = 10080;

//ID1A5B26F1-7CSNMXdE --> testrl1
//ID715C4B99-dSNcQnGe --> testrl02
//ID182C1F54-EaPm7ffa --> testrl03
std::vector<std::string> RlSetting::Names {"ID715C4B99-dSNcQnGe", "ID1A5B26F1-7CSNMXdE",
											"ID182C1F54-EaPm7ffa"};
//std::vector<std::string> RlSetting::Names {"ID5F706D6D-2WBML2Pe"}; //testrl0
}
