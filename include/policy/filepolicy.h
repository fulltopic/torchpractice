/*
 * filepolicy.h
 *
 *  Created on: Apr 19, 2020
 *      Author: zf
 */

#ifndef INCLUDE_POLICY_FILEPOLICY_H_
#define INCLUDE_POLICY_FILEPOLICY_H_

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <torch/torch.h>

#include "../utils/logger.h"
#include "policy/tenhoupolicy.h"

struct FilePolicyMsg {
	int type;
	std::string msg;
};

class FilePolicy: public TenhouPolicy {
private:
	const std::string path;
	std::vector<FilePolicyMsg> msgs;
	std::shared_ptr<spdlog::logger>& logger;
	int index;

public:
	FilePolicy(std::string iPath);
	void init();
	virtual ~FilePolicy();
	virtual int getAction(torch::Tensor values, const std::vector<int>& candidates);
	virtual int getAction(torch::Tensor values, const std::vector<int>& candidates, const std::vector<int>& excludes);
	virtual std::vector<int> getTiles4Action(torch::Tensor values, int actionType, const std::vector<int>& candidates, const int raw);
	virtual void reset();
};



#endif /* INCLUDE_POLICY_FILEPOLICY_H_ */
