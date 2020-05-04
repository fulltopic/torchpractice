/*
 * filepolicy.h
 *
 *  Created on: Apr 19, 2020
 *      Author: zf
 */

#ifndef INCLUDE_TENHOUCLIENT_FILEPOLICY_H_
#define INCLUDE_TENHOUCLIENT_FILEPOLICY_H_

#include "tenhoupolicy.h"
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include "logger.h"

struct FilePolicyMsg {
	int type;
	std::string msg;
};

class FilePolicy: public TenhouPolicy {
private:
	const std::string path;
	std::vector<FilePolicyMsg> msgs;
	std::shared_ptr<spdlog::logger> logger;
	int index;

public:
	FilePolicy(std::string iPath);
	void init();
	virtual ~FilePolicy();
	virtual int getAction(torch::Tensor values, std::vector<int> candidates) ;
	virtual int getAction(torch::Tensor values, std::vector<int> candidates, std::vector<int> excludes);
	virtual std::vector<int> getTiles4Action(torch::Tensor values, int actionType, std::vector<int> candidates);
	virtual void reset();
};



#endif /* INCLUDE_TENHOUCLIENT_FILEPOLICY_H_ */
