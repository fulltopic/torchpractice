/*
 * teststats.cpp
 *
 *  Created on: Sep 30, 2020
 *      Author: zf
 */


#include "utils/teststats.h"
#include <string>
#include <ctime>
#include <chrono>

RlTestStatData::RlTestStatData():
	clientId(""), seq(0), endType(InvalidGameEnd), who(-1), fromWho(-1), reward(0) {

}

RlTestStatData::RlTestStatData(std::string id, uint32_t gameSeq, GameEndType type, int winnerIndex, int fromWhoIndex, float r)
	: clientId(id), seq(gameSeq), endType(type), who(winnerIndex), fromWho(fromWhoIndex), reward(r)
{
}

std::ostream& operator<<(std::ostream& os, const RlTestStatData& data) {
	os << data.clientId << ", "
			<< std::to_string(data.seq) << ", "
			<< std::to_string(data.endType) << ", "
			<< std::to_string(data.who) << ", "
			<< std::to_string(data.fromWho) << ", "
			<< std::to_string(data.reward);

	return os;
}

//TODO: Switch file name per day
RlTestStatRecorder::RlTestStatRecorder(const std::string outputFileName)
	: fileName(outputFileName)
//, output(outputFileName)
{
	auto createTime = std::chrono::system_clock::now().time_since_epoch();
	auto createSecond = std::chrono::duration_cast<std::chrono::seconds>(createTime).count();
	lastSaveHour = std::chrono::duration_cast<std::chrono::hours>(createTime);
	std::string saveFileName = fileName + "_" + std::to_string(createSecond) + ".txt";
	output = std::ofstream(saveFileName);
}

RlTestStatRecorder::~RlTestStatRecorder() {
	output.close();
}

std::shared_ptr<RlTestStatRecorder> RlTestStatRecorder::GetRecorder(const std::string fileName) {
//	return std::make_shared<RlTestStatRecorder>(fileName); //TODO: This sentence failed to be compiled
	return std::shared_ptr<RlTestStatRecorder>(new RlTestStatRecorder(fileName));
}

bool RlTestStatRecorder::push(RlTestStatData&& data) {
	return dataQ.push(std::move(data));
}

void RlTestStatRecorder::write2File() {
	while (true) {
		while (dataQ.isEmpty()) {
			sleep(300); //TODO: remove magic number
		}

		while (!dataQ.isEmpty()) {
			auto data = dataQ.pop();
			output << data << std::endl;
		}
		output.flush();

		auto currTime = std::chrono::system_clock::now().time_since_epoch();
		auto currHour = std::chrono::duration_cast<std::chrono::hours>(currTime);
		if ((currHour.count() - lastSaveHour.count()) > 24) {
			auto fileSaveTime = std::chrono::duration_cast<std::chrono::seconds>(currTime).count();
			std::string saveFileName = fileName + "_" + std::to_string(fileSaveTime);
			output.close();
			output = std::ofstream(saveFileName);

			lastSaveHour = currHour;
		}
	}
}


