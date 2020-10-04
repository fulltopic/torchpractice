/*
 * rltestsetting.cpp
 *
 *  Created on: Sep 7, 2020
 *      Author: zf
 */


#include "rltest/rltestsetting.h"

namespace rltest {

const int RlSetting::ProxyNum = 4;
const int RlSetting::BatchSize = 16;
const int RlSetting::UpdateThreshold = 2; //TODO: dirty data

const float RlSetting::ReturnGamma = 0.9;

//const int RlSetting::NetNum = 2;
const int RlSetting::SaveEpochThreshold = 2;
const int RlSetting::ThreadNum = 4;

const std::string RlSetting::ServerIp = "133.242.10.78";
const int RlSetting::ServerPort = 10080;

const bool RlSetting::IsPrivateTest = false;

//ID1A5B26F1-7CSNMXdE --> testrl1
//ID715C4B99-dSNcQnGe --> testrl02
//ID182C1F54-EaPm7ffa --> testrl03
//ID57133897-hYBgTJ6h --> testrl04
//ID11751BFC-J53bNd7V --> testrl05
//"ID1A5B26F1-7CSNMXdE", invalid
std::vector<std::string> RlSetting::Names { "ID715C4B99-dSNcQnGe",
											"ID182C1F54-EaPm7ffa", "ID57133897-hYBgTJ6h",
											"ID11751BFC-J53bNd7V"};
/* {2, 4, 3, 8
 * }
 */

const std::string RlSetting::ModelDir = "/home/zf/workspaces/workspace_cpp/torchpractice/build/data/";
const std::string RlSetting::StatsDataName = "/home/zf/workspaces/workspace_cpp/torchpractice/build/data/statsdata";
//std::vector<std::string> RlSetting::Names {"ID5F706D6D-2WBML2Pe"}; //testrl0
}
