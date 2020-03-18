/*
 * Lmdb2RowDataDefs.cpp
 *
 *  Created on: Jan 24, 2020
 *      Author: zf
 */




#include "lmdbtools/Lmdb2RowDataDefs.h"

Lmdb2RowDataDefs::Lmdb2RowDataDefs() {}

Lmdb2RowDataDefs::~Lmdb2RowDataDefs() {}

const std::vector<int64_t> Lmdb2RowDataDefs::GetDataDim() {
	static std::vector<int64_t> dataDim{1, 2, 36};
	return dataDim;
}

const std::vector<int64_t> Lmdb2RowDataDefs::GetLabelDim() {
	static std::vector<int64_t> labelDim{1};
	return labelDim;
}
