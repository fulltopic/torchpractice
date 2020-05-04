/*
 * testbalance.cpp
 *
 *  Created on: Mar 26, 2020
 *      Author: zf
 */


#include <vector>
#include <map>

#include <torch/torch.h>
#include "lmdbtools/LmdbReader.h"
#include "lmdbtools/LmdbDataDefs.h"
#include "lmdbtools/Lmdb2RowDataDefs.h"
#include "lmdbtools/LmdbReaderWrapper.h"

const std::string dbPath = "/home/zf/workspaces/res/dbs/lmdbcpptest";

struct DictDataType {
	torch::Tensor tensor;
	int action;
};
const int DataLen = 5 * 72;
const int ClassNum = 42;

static std::size_t hashTensor(torch::Tensor tensor) {
	std::size_t seed = DataLen;
	auto t = tensor.data_ptr<float>();
	for (int i = 0; i < DataLen; i ++) {
		seed ^= (int)t[i] + 0x9e3779b9 + (seed << 6) + (seed >> 2);
	}

	return seed;
}

static bool tensorEq(const torch::Tensor& t1, const torch::Tensor& t2) {
	auto diff = torch::sub(t1, t2);
	auto diffSize = diff.nonzero().size(0);

	return diffSize == 0;
}

static void test() {
	LmdbSceneReader<LmdbDataDefs> reader(dbPath);
	std::map<std::size_t, std::vector<DictDataType>> dict;
	std::vector<std::vector<int>> errstats(ClassNum, std::vector<int>(ClassNum, 0));
	uint64_t totalConf = 0;
	uint64_t totalNum = 0;

	while (reader.hasNext()) {
		torch::Tensor datas;
		torch::Tensor labels;

		std::tie(datas, labels) = reader.next();

		for (int i = 0; i < datas.size(0); i ++) {
			totalNum ++;
			torch::Tensor data = datas[i];
			int label = (int)labels[i].item<long>();

			DictDataType dictData{data, label};
			std::size_t dataHash = hashTensor(data);

			if (dict.find(dataHash) == dict.end()) {
				std::vector<DictDataType> newData{dictData};
				dict[dataHash] = newData;
			} else {
				auto siblings = dict[dataHash];
				for (int j = 0; j < siblings.size(); j ++) {
					auto sibling = siblings[j];
					if (tensorEq(data, sibling.tensor) && (label != sibling.action)) {
						if (totalConf % 1000 == 0) {
							std::cout << "Get conflicted tensors " << std::endl;
							std::cout << "Tensor ----------------------------> " << sibling.action << std::endl;
							std::cout << sibling.tensor << std::endl;
							std::cout << "The other -------------------------> " << label << std::endl;
							std::cout << data << std::endl;
							std::cout << "End " << std::endl << std::endl;
						}

						totalConf ++;
						errstats[label][sibling.action] ++;
						errstats[sibling.action][label] ++;
					}
				}
				dict[dataHash].push_back(dictData);
			}
		}
	}

	std::cout << "Error stats: " << totalNum << std::endl;
	for (int i = 0; i < ClassNum; i ++) {
		for (int j = 0; j < ClassNum; j ++) {
			std::cout << errstats[i][j] << ", ";
		}
		std::cout << std::endl;
	}
}

int main() {
	test();
}
