/*
 * testtensor.cpp
 *
 *  Created on: May 10, 2020
 *      Author: zf
 */

#include <torch/torch.h>
#include <iostream>

using namespace std;
using namespace torch;

int testSort() {
	vector<int> candidates {0, 2, 5, 10, 12, 20, 21, 22, 25, 28};
	auto values = torch::rand({42});
	cout << "Values " << values << endl;

	auto rc = values.sort(0, true);
	cout << "Sorted values " << get<0>(rc) << endl;
	auto indexes = std::get<1>(rc);
	cout << "Sorted index " << indexes << endl;

	auto dataPtr = indexes.data_ptr<long>();
	for (int i = 0; i < indexes.numel(); i ++) {
		if (find(candidates.begin(), candidates.end(), (int)dataPtr[i]) != candidates.end()) {
			cout << "Found " << dataPtr[i] << endl;
			return (int)dataPtr[i];
		}
	}

	return -1;
}

int main() {
	cout << testSort() << endl;
}
