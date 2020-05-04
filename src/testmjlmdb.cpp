/*
 * testmjlmdb.cpp
 *
 *  Created on: Jan 2, 2020
 *      Author: zf
 */

#include "lmdbtools/LmdbReader.h"
#include "lmdbtools/LmdbDataDefs.h"
#include "lmdbtools/Lmdb2RowDataDefs.h"
#include <torch/torch.h>

int main() {
	const std::string cppCreateDb = "/home/zf/workspaces/res/dbs/lmdbscenetest";
//	LmdbDataDefs dataDefs;
	LmdbSceneReader<LmdbDataDefs> reader(cppCreateDb);

	const std::string cpp2RowCreateDb = "/home/zf/workspaces/res/dbs/lmdbscene2rowtest";
	LmdbSceneReader<Lmdb2RowDataDefs> denseReader(cpp2RowCreateDb);


//	while (reader.hasNext()) {
//		torch::Tensor data;
//		torch::Tensor label;
//		std::tie(data, label) = reader.next();
//
//		std::cout << data.sizes() << std::endl;
//		std::cout << label.sizes() << std::endl;
//	}

	while (denseReader.hasNext()) {
		torch::Tensor data;
		torch::Tensor label;
		std::tie(data, label) = denseReader.next();

		std::cout << data.sizes() << std::endl;
		std::cout << label.sizes() << std::endl;

//		break;
	}
}


