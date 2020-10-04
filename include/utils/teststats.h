/*
 * teststats.h
 *
 *  Created on: Sep 29, 2020
 *      Author: zf
 */

#ifndef INCLUDE_RLTEST_TESTSTATS_H_
#define INCLUDE_RLTEST_TESTSTATS_H_

#include <vector>
#include <cinttypes>
#include <string>
#include <memory>
#include <iostream>
#include <fstream>
#include <ctime>
#include <chrono>

#include "dataqueue.hpp"

enum GameEndType {
	AGA = 0, //AGARI
	RYU = 1, //RYUUKYOKU
	InvalidGameEnd = 2,
};

struct RlTestStatData {
public:
std::string clientId;
uint32_t seq;
GameEndType endType;
int who;
int fromWho;
float reward;

RlTestStatData();
//TODO: Why std::string& is invalid
RlTestStatData(std::string id, uint32_t gameSeq, GameEndType type, int winnerIndex, int fromWhoIndex, float r);

friend std::ostream& operator<<(std::ostream& os, const RlTestStatData& data);
};
std::ostream& operator<<(std::ostream& os, const RlTestStatData& data);


class RlTestStatRecorder {
private:
	const std::string fileName;
	std::ofstream output;
	std::chrono::hours lastSaveHour;
	R1WmQueue<RlTestStatData, 256> dataQ;


	RlTestStatRecorder(const std::string outputFileName);
	//iostream of file.
	//TODO: How to deal with exception in constructor?

public:
	~RlTestStatRecorder();

	static std::shared_ptr<RlTestStatRecorder> GetRecorder(const std::string fileName);
	bool push(RlTestStatData&& data); //Need a copy
	void write2File();
};





#endif /* INCLUDE_RLTEST_TESTSTATS_H_ */
