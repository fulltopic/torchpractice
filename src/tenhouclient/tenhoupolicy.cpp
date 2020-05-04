/*
 * tenhoupolicy.cpp
 *
 *  Created on: Apr 17, 2020
 *      Author: zf
 */


//#include "tenhouclient/netproxy.h"
#include "tenhouclient/tenhoumsgparser.h"
#include "tenhouclient/tenhoumsggenerator.h"
#include "tenhouclient/tenhouconsts.h"
#include "tenhouclient/netproxy.h"
//#include "tenhoufsm.h"
//
//#include "nets/a3cnet.h"

#include <vector>
#include <string>

#include <torch/torch.h>

using namespace std;
using namespace torch;

TenhouPolicy::~TenhouPolicy() {}

RandomPolicy::RandomPolicy(float rate): rndRate(rate)
{
}

int RandomPolicy::getAction(Tensor values, vector<int> candidates) {
//	values.argmax(0);
	auto rc = values.sort(0, true);
	auto indexes = std::get<1>(rc);

	auto dataPtr = indexes.data_ptr<float>();
	for (int i = 0; i < indexes.numel(); i ++) {
		if (find(candidates.begin(), candidates.end(), (int)dataPtr[i]) != candidates.end()) {
			return (int)dataPtr[i];
		}
	}

	return -1;
}

int RandomPolicy::getAction(Tensor values, vector<int> candidates, std::vector<int> excludes) {
	auto rc = values.sort(0, true);
	auto indexes = std::get<1>(rc);

	auto dataPtr = indexes.data_ptr<float>();
	for (int i = 0; i < indexes.numel(); i ++) {
		if (find(candidates.begin(), candidates.end(), (int)dataPtr[i]) != candidates.end()) {
			if (find(excludes.begin(), excludes.end(), (int)dataPtr[i]) == excludes.end()) {
				return (int)dataPtr[i];
			}
		}
	}

	return -1;
}
