/*
 * dirpolicy.h
 *
 *  Created on: May 4, 2020
 *      Author: zf
 */

#ifndef INCLUDE_POLICY_DIRPOLICY_H_
#define INCLUDE_POLICY_DIRPOLICY_H_

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <torch/torch.h>

#include "../utils/logger.h"
#include "policy/tenhoupolicy.h"

struct DirPolicyMsg {
	int type;
	std::string msg;
};

class DirPolicy: public TenhouPolicy {
private:
	const std::string path;
	std::vector<DirPolicyMsg> msgs;
	std::shared_ptr<spdlog::logger> logger;
	int index;
	int fileIndex;
	std::vector<std::string> fileNames;

	std::vector<DirPolicyMsg> parseTenhouFile(std::string fileName);

public:
	DirPolicy(std::string iPath);
	void init();
	virtual ~DirPolicy();
	virtual int getAction(torch::Tensor values, const std::vector<int>& candidates);
	virtual int getAction(torch::Tensor values, const std::vector<int>& candidates, const std::vector<int>& excludes);
	virtual std::vector<int> getTiles4Action(torch::Tensor values, int actionType, const std::vector<int>& candidates, const int raw) ;
	virtual void reset();
};



#endif /* INCLUDE_POLICY_DIRPOLICY_H_ */
