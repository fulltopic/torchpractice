/*
 * LmdbDataDefs.h
 *
 *  Created on: Jan 3, 2020
 *      Author: zf
 */

#ifndef INCLUDE_LMDBTOOLS_LMDBDATADEFS_H_
#define INCLUDE_LMDBTOOLS_LMDBDATADEFS_H_

//#include "DataDefs.h"
#include <vector>
#include <bits/stdint-intn.h>

class LmdbDataDefs {
protected:
	LmdbDataDefs();
	~LmdbDataDefs();
public:
	static const std::vector<int64_t> GetDataDim();
	static const std::vector<int64_t> GetLabelDim();
};



#endif /* INCLUDE_LMDBTOOLS_LMDBDATADEFS_H_ */
