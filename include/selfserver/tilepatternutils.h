/*
 * tilepatternutils.h
 *
 *  Created on: Oct 29, 2020
 *      Author: zf
 */

#ifndef INCLUDE_SELFSERVER_TILEPATTERNUTILS_H_
#define INCLUDE_SELFSERVER_TILEPATTERNUTILS_H_

#include <string>
#include <vector>

class TilePatternUtils {
public:
	enum {
		TilePerMajor = 4,
		PlayerNum = 4,
		TotalTileNum = 136,
		InitTen = 80,
		NormTile = 13,
		DrawRemain = 14, //Ryu
		Mangan = 2000,
	};

	static int GetMajorNum (const std::vector<int>& tiles);
	static int GetMsgIndex (int index, int myIndex);
	static std::string GenScMsg (int index, const std::vector<int>& tens, const std::vector<int>& gains);
	static std::string IntVec2Str (std::string head, const std::vector<int>& data);

	static bool IsKyushu(const std::vector<int>& tiles);
	static bool IsOrphan(const std::vector<int>& tiles);

	static std::vector<std::string> GenOrphanMsg(int who, const std::vector<int>& tens, const std::vector<int>& tiles);
	static std::string GenProfMsg();
	static std::vector<std::string> Gen9YaoMsg(int index, std::vector<int>& tiles, std::vector<int>& tens);
};



#endif /* INCLUDE_SELFSERVER_TILEPATTERNUTILS_H_ */
