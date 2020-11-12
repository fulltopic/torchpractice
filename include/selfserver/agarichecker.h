/*
 * AgariChecker.h
 *
 *  Created on: Oct 31, 2020
 *      Author: zf
 */

#ifndef INCLUDE_SELFSERVER_AGARICHECKER_H_
#define INCLUDE_SELFSERVER_AGARICHECKER_H_

#include <vector>
#include <map>
#include <memory>

class AgariChecker {
private:
	std::map<int, std::vector<int>> table;
	void init1();
	void init2();
	void init3();
	void init4();

//	static std::vector<int> TileCounts(std::vector<int>& tiles) ;

	AgariChecker();

public:
//	void Init();
	std::pair<int, std::vector<int>> getKey(std::vector<int>& n);
	bool isAgari(int key);
	std::vector<int> getCombs (int key);

	static std::shared_ptr<AgariChecker>& GetInst();
};



#endif /* INCLUDE_SELFSERVER_AGARICHECKER_H_ */

