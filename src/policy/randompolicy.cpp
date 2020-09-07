/*
 * randompolicy.cpp
 *
 *  Created on: May 9, 2020
 *      Author: zf
 */


#include <vector>
#include <string>
#include <set>
#include <iostream>
#include <stdlib.h>
#include <time.h>

#include <torch/torch.h>

#include "policy/randompolicy.h"
#include "tenhouclient/tenhoustate.h"
#include "tenhouclient/tenhoumsgparser.h"

using namespace std;
using namespace torch;

using P = TenhouMsgParser;

RandomPolicy::RandomPolicy(float rate): rndRate(rate), logger(Logger::GetLogger())
{
	std::srand(time(nullptr));
}

RandomPolicy::~RandomPolicy() {}

int RandomPolicy::getAction(Tensor values, vector<int> candidates) {
	auto rc = values.sort(-1, true);
	auto indexes = std::get<1>(rc);
//	cout << "indexes " << indexes << endl;
//	cout << "sorted values: " << std::get<0>(rc) << endl;

	if ((rand() % 100) > (rndRate * 100)) {
//	std::cout << "Get values sizes " << values.sizes() << std::endl;
//
		if (find(candidates.begin(), candidates.end(), LstmStateAction::ChowAction) != candidates.end()) {
			return LstmStateAction::ChowAction;
		}
		if (find(candidates.begin(), candidates.end(), LstmStateAction::AnKanAction) != candidates.end()) {
			return LstmStateAction::AnKanAction;
		}
		if (find(candidates.begin(), candidates.end(), LstmStateAction::KaKanAction) != candidates.end()) {
			return LstmStateAction::KaKanAction;
		}
		if (find(candidates.begin(), candidates.end(), LstmStateAction::MinKanAction) != candidates.end()) {
			return LstmStateAction::MinKanAction;
		}
		if (find(candidates.begin(), candidates.end(), LstmStateAction::PongAction) != candidates.end()) {
			return LstmStateAction::PongAction;
		}
		if (find(candidates.begin(), candidates.end(), LstmStateAction::ReachAction) != candidates.end()) {
			return LstmStateAction::ReachAction;
		}
		if (find(candidates.begin(), candidates.end(), LstmStateAction::RonAction) != candidates.end()) {
			return LstmStateAction::RonAction;
		}
	}
//	values.argmax(0);

	auto dataPtr = indexes.data_ptr<long>();
	for (int i = 0; i < indexes.numel(); i ++) {
		if (find(candidates.begin(), candidates.end(), (int)dataPtr[i]) != candidates.end()) {
			return (int)dataPtr[i];
		}
	}

	return -1;
}

int RandomPolicy::getAction(Tensor values, vector<int> candidates, std::vector<int> excludes) {
//	if (find(candidates.begin(), candidates.end(), LstmStateAction::ChowAction) != candidates.end()) {
//		return LstmStateAction::ChowAction;
//	}
//	if (find(candidates.begin(), candidates.end(), LstmStateAction::AnKanAction) != candidates.end()) {
//		return LstmStateAction::AnKanAction;
//	}
//	if (find(candidates.begin(), candidates.end(), LstmStateAction::KaKanAction) != candidates.end()) {
//		return LstmStateAction::KaKanAction;
//	}
//	if (find(candidates.begin(), candidates.end(), LstmStateAction::MinKanAction) != candidates.end()) {
//		return LstmStateAction::MinKanAction;
//	}
//	if (find(candidates.begin(), candidates.end(), LstmStateAction::PongAction) != candidates.end()) {
//		return LstmStateAction::PongAction;
//	}
//	if (find(candidates.begin(), candidates.end(), LstmStateAction::ReachAction) != candidates.end()) {
//		return LstmStateAction::ReachAction;
//	}
//	if (find(candidates.begin(), candidates.end(), LstmStateAction::RonAction) != candidates.end()) {
//		return LstmStateAction::RonAction;
//	}
//
//	auto rc = values.sort(0, true);
//	auto indexes = std::get<1>(rc);
//
//	auto dataPtr = indexes.data_ptr<float>();
//	for (int i = 0; i < indexes.numel(); i ++) {
//		if (find(candidates.begin(), candidates.end(), (int)dataPtr[i]) != candidates.end()) {
//			if (find(excludes.begin(), excludes.end(), (int)dataPtr[i]) == excludes.end()) {
//				return (int)dataPtr[i];
//			}
//		}
//	}
//
//	return -1;
	return getAction(values, candidates);
}

vector<int> RandomPolicy::getTiles4Action(Tensor values, int actionType, vector<int> candidates, const int raw) {
	logger->debug("Get tiles for action {}", actionType);

	if (actionType != LstmStateAction::ChowAction) {
		return candidates;
	}

	if (candidates.size() <= 2) {
		return candidates;
	}

	const int rawTile = P::Raw2Tile(raw);

	vector<int> poses;
	vector<int> tiles;
	vector<float> intentions;
	auto dataPtr = values.accessor<float, 2>();
	int lastTile = -1;
	for (int i = 0; i < candidates.size(); i ++) {
		int tile = P::Raw2Tile(candidates[i]);
		if (tile == rawTile) {
			continue;
		}
		if (tile != lastTile) {
			poses.push_back(i);
			tiles.push_back(tile);
			intentions.push_back(dataPtr[0][tile]);
			lastTile = tile;
		}
	}

	if (tiles.size() < 2) {
		logger->error("Error in candidates ");
		return vector<int>();
	} else if (tiles.size() == 2) {
		return {candidates[poses[0]], candidates[poses[1]]};
	}

	//candidates sorted
	int tileIndex1 = -1;
	int tileIndex2 = -1;
	float minIntention = 2;

	for (int i = 0; i < tiles.size() - 1; i ++) {
		float intention = intentions[i] + intentions[i + 1];

		if (((tiles[i] == rawTile - 2) && (tiles[i + 1] == rawTile - 1))
				||((tiles[i] == rawTile - 1) && (tiles[i] == rawTile + 1))
				||((tiles[i] == rawTile + 1) && (tiles[i + 1] == rawTile + 2))
			)
		{
			logger->debug("Get tiles for chow {}, {}", tiles[i], tiles[i + 1]);
			if (intention < minIntention) {
				tileIndex1 = i;
				tileIndex2 = i + 1;
				minIntention = intention;
			}

		}
	}

	if (tileIndex1 >= 0) {
		logger->debug("Return {}, {}", candidates[poses[tileIndex1]], candidates[poses[tileIndex2]]);
		return {candidates[poses[tileIndex1]], candidates[poses[tileIndex2]]};
	} else {
		logger->error("Failed to get chow tiles ");
		return {};
	}
}

void RandomPolicy::reset() {
	//nothing
}


