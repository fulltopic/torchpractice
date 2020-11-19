/*
 * rltestsetting.cpp
 *
 *  Created on: Sep 7, 2020
 *      Author: zf
 */


#include "rltest/rltestsetting.h"

namespace rltest {

const int RlSetting::ProxyNum = 1;
const int RlSetting::ThreadNum = 1;
const int RlSetting::BatchSize = 8;
const int RlSetting::UpdateThreshold = 64; //TODO: update network in time

const float RlSetting::ReturnGamma = 0.9; //TODO: Refer paper for proper gamma
const float RlSetting::RewardClip = 100;

//const int RlSetting::NetNum = 2;
const int RlSetting::SaveEpochThreshold = 1;

//const std::string RlSetting::ServerIp = "133.242.10.78";
//const int RlSetting::ServerPort = 10080;
const std::string RlSetting::ServerIp = "127.0.0.1";
const int RlSetting::ServerPort = 55555;

const bool RlSetting::IsPrivateTest = false;
const bool RlSetting::IsTest = false;

//ID1A5B26F1-7CSNMXdE --> testrl1
//ID715C4B99-dSNcQnGe --> testrl02 --> lost (ERR1003)
//ID182C1F54-EaPm7ffa --> testrl03
//ID57133897-hYBgTJ6h --> testrl04
//ID11751BFC-J53bNd7V --> testrl05
//ID7F3A4211-GDbZHRLN --> testrl06
//ID20616B71-8XMESen8 --> testrl07
//ID0EA3406B-MQ542M27 --> testrl08
//ID60911A97-SYV9F368 --> testrl09
//ID17EB2588-JZfWWTF7 --> testrl10
//"ID1A5B26F1-7CSNMXdE", invalid
std::vector<std::string> RlSetting::Names { "ID182C1F54-EaPm7ffa",
											"ID57133897-hYBgTJ6h",
											"ID11751BFC-J53bNd7V",
											"ID7F3A4211-GDbZHRLN",
											"ID20616B71-8XMESen8",
											"ID0EA3406B-MQ542M27",
											"ID60911A97-SYV9F368",
											"ID17EB2588-JZfWWTF7"
};
/* {2, 4, 3, 8
 * }
 */

const std::string RlSetting::ModelDir = "/home/zf/workspaces/workspace_cpp/torchpractice/build/data/";
const std::string RlSetting::StatsDataName = "/home/zf/workspaces/workspace_cpp/torchpractice/build/data/statsdata";
const std::string RlSetting::LossStateName = "/home/zf/workspaces/workspace_cpp/torchpractice/build/data/lossstatdata";
//std::vector<std::string> RlSetting::Names {"ID5F706D6D-2WBML2Pe"}; //testrl0
}
