/*
 * tenhouproxy.h
 *
 *  Created on: Apr 13, 2020
 *      Author: zf
 */

#ifndef INCLUDE_TENHOUCLIENT_NETPROXY_H_
#define INCLUDE_TENHOUCLIENT_NETPROXY_H_

#include "tenhouconsts.h"
#include "tenhoupolicy.h"
#include "tenhoustate.h"
#include "logger.h"

#include "randomnet.h"

#include <vector>
#include <string>

#include <torch/torch.h>



class RandomPolicy: public TenhouPolicy {
private:
	float rndRate;
public:
	RandomPolicy(float rate);
	virtual ~RandomPolicy() = default;

	virtual int getAction(torch::Tensor values, std::vector<int> candidates);
	virtual int getAction(torch::Tensor values, std::vector<int> candidates, std::vector<int> excludes);
};

class NetProxy {
private:
//	torch::Tensor board;
	TenhouState& innerState;

	torch::Tensor rnnHidden;
	torch::Tensor rnnState;

	RandomNet net;
	TenhouPolicy& policy;

	std::shared_ptr<spdlog::logger> logger;

	std::string processInitMsg(std::string msg);
	std::string processDoraMsg(std::string msg);
	std::string processDropMsg(std::string msg);
	std::string processAccept(std::string msg);
	std::string processNMsg(std::string msg);
	std::string processReachMsg(std::string msg);
	std::string processIndicatorMsg(std::string msg);
	std::string processReachInd(int raw);
	std::string processChowInd(int fromWho, int raw);
	std::string processPongInd(int fromWho, int raw);
	std::string processKanInd(int fromWho, int raw);
	std::string processPongKanInd(int fromWho, int raw);
	std::string processPongChowInd(int fromWho, int raw);
	std::string processPongChowKanInd(int fromWho, int raw);
	std::string processRonInd(int fromWho, int raw, int type, bool isTsumogiri);
	std::string processGameEndInd(std::string msg);

public:
	NetProxy(TenhouState& state, TenhouPolicy& iPolicy);
	~NetProxy() = default;
	std::string processMsg(std::string msg);
	void reset();
};

#endif /* INCLUDE_TENHOUCLIENT_NETPROXY_H_ */
