/*
 * tenhouconsts.h
 *
 *  Created on: Apr 10, 2020
 *      Author: zf
 */

#ifndef INCLUDE_TENHOUCLIENT_TENHOUCONSTS_H_
#define INCLUDE_TENHOUCLIENT_TENHOUCONSTS_H_

enum TenhouConsts {
	ME = 0,
	PlayerNum = 4,
	NumPerCategory = 9,
	NumPerTile = 4,
	UsualTileNum = 13,
	TileNum = 34,

	ChowFlag = 1 << 2,
	PongFlag = 1 << 3,
	KakanFlag = 1 << 4,
	KitaBits = 5,
	AnkanFlag = 3,
	MinkanFlag = 1 << 5,
	KanFlag = 1 << 6,
	InvalidFlag = 0,
};



#endif /* INCLUDE_TENHOUCLIENT_TENHOUCONSTS_H_ */
