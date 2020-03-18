/*
 * LmdbDataDefs.cpp
 *
 *  Created on: Jan 3, 2020
 *      Author: zf
 */



#include "lmdbtools/LmdbDataDefs.h"

LmdbDataDefs::LmdbDataDefs() {}

LmdbDataDefs::~LmdbDataDefs() {}

const std::vector<int64_t> LmdbDataDefs::GetDataDim() {
	static std::vector<int64_t> dataDim{1, 5, 72};
	return dataDim;
}

const std::vector<int64_t> LmdbDataDefs::GetLabelDim() {
	static std::vector<int64_t> labelDim{1};
	return labelDim;
}
