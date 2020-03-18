/*
 * Lmdb2RowDataDefs.h
 *
 *  Created on: Jan 24, 2020
 *      Author: zf
 */

#ifndef INCLUDE_LMDBTOOLS_LMDB2ROWDATADEFS_H_
#define INCLUDE_LMDBTOOLS_LMDB2ROWDATADEFS_H_

#include <vector>
#include <bits/stdint-intn.h>

class Lmdb2RowDataDefs {
protected:
	Lmdb2RowDataDefs();
	~Lmdb2RowDataDefs();
public:
	static const std::vector<int64_t> GetDataDim();
	static const std::vector<int64_t> GetLabelDim();
};



#endif /* INCLUDE_LMDBTOOLS_LMDB2ROWDATADEFS_H_ */
