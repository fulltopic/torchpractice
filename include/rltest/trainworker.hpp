/*
 * trainworker.h
 *
 *  Created on: Sep 6, 2020
 *      Author: zf
 */

#ifndef INCLUDE_RLTEST_TRAINWORKER_HPP_
#define INCLUDE_RLTEST_TRAINWORKER_HPP_

#include <vector>
#include <queue>


#include "tenhouclient/netproxy.hpp"

#include "returncal.h"
#include "rltestsetting.h"

template <class NetType>
class RlTrainWorker {
using StoreDataType = std::vector<std::vector<torch::Tensor>>;

private:
	int trainingNetIndex;
	int trainEpoch;

	ReturnCalculator& calc;

	//TODO: inject innstate and policy
	std::vector<NetProxy<NetType>> netProxies;

	std::queue<StoreDataType> dataQ;

	std::vector<std::shared_ptr<NetType>> nets;
	torch::optim::Optimizer& optimizer;
	//TODO: optimizer

	std::shared_ptr<spdlog::logger> logger;

	void trainingWork();

	void trainingNet();
public:
	//TODO: TenhouPolicy is stateless
	template<typename... Args>
	RlTrainWorker(ReturnCalculator& calculator, torch::optim::Optimizer& opter, TenhouPolicy& policy, Args... args);
	~RlTrainWorker();
	RlTrainWorker(const RlTrainWorker& other) = delete;
	RlTrainWorker& operator=(const RlTrainWorker& other) = delete;
	//TODO: move constructors


	void start();
};

template<class NetType>
template<typename... Args>
RlTrainWorker<NetType>::RlTrainWorker<Args>(ReturnCalculator& iCalculator, torch::optim::Optimizer& opter,
		TenhouPolicy& iPolicy, Args... args):
	trainingNetIndex(0),
	trainEpoch(0),
	calc(iCalculator),
	optimizer(opter),
	logger(Logger::GetLogger())

{
	for (int i = 0; i < rltest::RlSetting::NetNum; i ++) {
		nets.push_back(std::shared_ptr<NetType>(new NetType(args)));
	}

	for (int i = 0; i < rltest::RlSetting::ProxyNum; i ++) {
		//TODO: Make innState configurable
		netProxies.push_back(NetProxy<NetType>(nets[0], BaseState(72, 5), iPolicy));
	}
}

template<class NetType>
void RlTrainWorker<NetType>::trainingNet() {
	if (dataQ.size() < rltest::RlSetting::BatchSize) {
		logger->debug("Not enough data to be trained");
		return;
	}

	std::vector<torch::Tensor> inputs;
//	std::vector<torch::Tensor> labels;

	for (int i = 0; i < rltest::RlSetting::BatchSize; i ++) {
		auto datas = dataQ.front();
		dataQ.pop();

		std::vector<torch::Tensor> inputVec;
		inputVec.reserve(datas.size());
		for (int j = 0; j < datas.size(); j ++) {
			inputVec.push_back(datas[j][StorageIndex::InputIndex]);
		}

		torch::Tensor input = torch::stack(inputVec, 0);
		inputs.push_back(input);
	}

	auto output = nets[trainingNetIndex]->forward(inputs, true);

}


#endif /* INCLUDE_RLTEST_TRAINWORKER_HPP_ */
