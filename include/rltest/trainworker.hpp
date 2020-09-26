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
#include <future>
#include <algorithm>
#include <iostream>
#include <iosfwd>
#include <sstream>
#include <memory>

#include <boost/asio.hpp>
#include <boost/thread/thread.hpp>
#include <boost/bind/bind.hpp>

#include "policy/randompolicy.h"

#include "rltest/l2net.h"

#include "tenhouclient/asiotenhoufsm.hpp"
#include "tenhouclient/tenhouconsts.h"
#include "tenhouclient/tenhoufsm.h"
#include "tenhouclient/tenhoufsmstate.h"
#include "tenhouclient/tenhoustate.h"
#include "tenhouclient/netproxy.hpp"

#include "utils/datastorequeue.h"

#include "returncal.h"
#include "rltestsetting.h"

template <class NetType, class OptType, class OptOptionType>
class TrainObj {
private:
	int curIndex;
	int lastIndex;
	std::shared_ptr<spdlog::logger> logger;

public:
//	static const int Cap;
	enum {Cap = 2,};
	std::vector<std::shared_ptr<NetType>> nets;
	std::vector<std::unique_ptr<OptType>> optimizers;


//	template<typename ...NetArgs>
//	explicit TrainObj(NetArgs args): curIndex(0), lastIndex(1) {
//		nets.push_back(std::shared_ptr<NetType>(args));
//
//		for (int i = 1; i < Cap; i ++) {
//			auto copy = nets[curIndex]->clone();
//			std::shared_ptr<NetType> cpyNet = std::dynamic_pointer_cast<NetType>(copy);
//			nets.push_back(cpyNet);
//		}
//	}
	explicit TrainObj(): curIndex(0), lastIndex(1), logger(Logger::GetLogger()) {
	}

	//TODO: Other constructors
	TrainObj(const TrainObj<NetType, OptType, OptOptionType>& other) = delete;

	template<typename ...NetArgs>
	void createNets(NetArgs&&... args) {
		nets.push_back(std::shared_ptr<NetType>(new NetType(args...)));
		logger->warn("Create original net");

		for (int i = 1; i < Cap; i ++) {
			auto copy = nets[0]->clone();
			std::shared_ptr<NetType> cpyNet = std::dynamic_pointer_cast<NetType>(copy);
			nets.push_back(cpyNet);

			logger->warn("Cloned double net");
		}
	}

	template<typename ...OptArgs>
	void createOptimizers(OptArgs&&... args) {
		for (int i = 0; i < Cap; i ++) {
			//TODO: move?
//			optimizers.push_back(OptType(nets[i]->parameters(), args...));
			optimizers.push_back(std::make_unique<OptType>(nets[i]->parameters(), args...));
			logger->warn("Create optimizer {}", i);
		}
	}

	//working network for agents
	std::shared_ptr<NetType> getWorkingNet() {
		logger->warn("Get current working(not training) net {}", lastIndex);
		return nets[lastIndex];
	}

	//network that is being updated
	std::shared_ptr<NetType> getTrainingNet() {
		return nets[curIndex];
	}

	void updateNet(std::vector<std::vector<torch::Tensor>> inputs) {
		logger->warn("Get loss of nets {}", curIndex);
		torch::Tensor loss = nets[curIndex]->getLoss(inputs);

		logger->warn("Optimizer by optimizer {}", curIndex);
		optimizers[curIndex]->zero_grad();
		loss.backward();
		optimizers[curIndex]->step();

		logger->warn("Worker set Net {} updated", curIndex);
	}

//	void cloneOpt(OptType& opt0, OptType& opt1, NetType& net0, NetType& net1) {
//		auto namedParams0 = net0->named_parameters(true);
//		auto namedParams1 = net1->named_parameters(true);
//		auto& states0 = opt0.state();
//		auto& states1 = opt1.state();
//		states1.clear();
//
//		for (auto ite = namedParams0.begin(); ite != namedParams0.end(); ite ++) {
//			auto key = ite->key();
//			logger->info("To deal with param {}", key);
//
//			torch::Tensor value0 = ite->value();
//			auto value0Str = c10::guts::to_string(value0.unsafeGetTensorImpl());
//
//			if (states0.find(value0Str) == states0.end()) {
//				logger->info("No state defined for param {}", key);
//				continue;
//			}
//
//			auto& currState = states0.at(value0Str);
//			auto stateClone = currState->clone();
//			torch::Tensor value1 = namedParams1[key];
//			auto value1Str = c10::guts::to_string(value1.unsafeGetTensorImpl());
//			states1[value1Str] = std::move(stateClone);
//		}
//	}
//	void cloneOpt(OptType& optFrom, OptType& optTo) {
//		std::ostringstream buf;
//		torch::serialize::OutputArchive output_archive;
//		optFrom.save(output_archive);
//		output_archive.save_to(buf);
//
//		std::istringstream inStream(buf.str());
//		torch::serialize::InputArchive input_archive;
//		input_archive.load_from(inStream);
//		optTo.load(input_archive);
//	}

	void switchNet() {
		auto copy = nets[curIndex]->clone();
		std::shared_ptr<NetType> cpyNet = std::dynamic_pointer_cast<NetType>(copy);
		nets[lastIndex] = cpyNet;
		logger->warn("Cloned net {} into net {}", curIndex, lastIndex);

		//TODO: Check if it is not rmspro optimizer
//		auto& options = nets[curIndex]->defaults();
//		optimizers[lastIndex] = OptType(nets[lastIndex]->parameters(), stat)
//		optimizers[lastIndex] = OptType(nets[lastIndex]->parameters());//, //clone options);
//		auto& paramGroups = optimizers[lastIndex].param_groups();
//		paramGroups.clear();
//		optimizers[lastIndex].add_param_group(cpyNet->parameters());
//		logger->warn("Update parameters bound to net {}", lastIndex);
		auto& optOptions = optimizers[curIndex]->defaults();
		optimizers[lastIndex] = std::move(std::make_unique<OptType>(nets[lastIndex]->parameters(), static_cast<OptOptionType&>(optOptions)));
		std::ostringstream buf;
		torch::serialize::OutputArchive output_archive;
		optimizers[curIndex]->save(output_archive);
		output_archive.save_to(buf);

		std::istringstream inStream(buf.str());
		torch::serialize::InputArchive input_archive;
		input_archive.load_from(inStream);
		optimizers[lastIndex]->load(input_archive);
//		cloneOpt(optimizers[curIndex], optimizers[lastIndex]);
		logger->warn("Optimizer updated ");

		int tmp = curIndex;
		curIndex = lastIndex;
		lastIndex = tmp;
		logger->warn("Switched. curIndex = {}, lastIndex = {}", curIndex, lastIndex);
	}
};

//template<class NetType, class OptType, class OptOptionType>
//const int TrainObj<NetType, OptType, OptOptionType>::Cap = 2;

template <class NetType, class OptType, class OptOptionType>
class RlTrainWorker {
using StoreDataType = std::vector<std::vector<torch::Tensor>>;

private:
	int trainEpoch;

	rltest::ReturnCalculator& calc;

	//TODO: inject innstate and policy
	std::vector<std::shared_ptr<NetProxy<NetType>>> netProxies;

	std::queue<StoreDataType> dataQ;

	TrainObj<NetType, OptType, OptOptionType>& trainingProxy;

	std::shared_ptr<spdlog::logger> logger;

	void trainingWork();

	bool trainingNet();
public:
	//TODO: TenhouPolicy is stateless
	//TODO: Remove reference(&) of input arguments (shared_ptr or membership)
	RlTrainWorker(rltest::ReturnCalculator& calculator, TenhouPolicy& policy, TrainObj<NetType, OptType, OptOptionType>& trainObj);
	~RlTrainWorker() {};
	RlTrainWorker(const RlTrainWorker& other) = delete;
	RlTrainWorker& operator=(const RlTrainWorker& other) = delete;
	//TODO: move constructors
	template<typename... Args>
	void createOptimizers(Args... args);

	//Network loaded outside of worker
	void start();
};

template<class NetType, class OptType, class OptOptionType>
RlTrainWorker<NetType, OptType, OptOptionType>::RlTrainWorker(rltest::ReturnCalculator& iCalculator,
		TenhouPolicy& iPolicy, TrainObj<NetType, OptType, OptOptionType>& trainObj):
	trainEpoch(0),
	calc(iCalculator),
	logger(Logger::GetLogger()),
	trainingProxy(trainObj)

{
}

//output of dataQ(datas) vector<vector<Tensor>>
//{vector<input>, vector<HState>, vector<Labels>, vector<Actions>, vector<Reward>}
//vector<input> -> {seqLen, input}
//vector<Reward> -> {reward}
template <class NetType, class OptType, class OptOptionType>
bool RlTrainWorker<NetType, OptType, OptOptionType>::trainingNet() {
	logger->warn("Get datasize: {}", dataQ.size());

	if (dataQ.size() < rltest::RlSetting::BatchSize) {
		logger->warn("Not enough data to be trained");
		return false;
	}

	std::vector<torch::Tensor> inputs;
	std::vector<torch::Tensor> labels;
	std::vector<torch::Tensor> actions;
	std::vector<torch::Tensor> rewards;

	while(!dataQ.empty()) {
		auto datas = dataQ.front();
		dataQ.pop();

		std::vector<torch::Tensor> inputData = datas[rltest::InputIndex];
		if (inputData.size() == 0) {
			logger->warn("popped invalid data");
			continue;
		}
		torch::Tensor input = torch::cat(inputData, 0); // {seqLen, 5, 72}
		inputs.push_back(input);

		std::vector<torch::Tensor> actionData = datas[rltest::ActionIndex];
		torch::Tensor action = torch::cat(actionData, 0);
		actions.push_back(action);

		std::vector<torch::Tensor> labelData = datas[rltest::LabelIndex];
		torch::Tensor label = torch::stack(labelData, 0);
		labels.push_back(label);

		torch::Tensor reward = datas[rltest::RewardIndex][0];
		rewards.push_back(reward);
	}
	if (rewards.size() == 0) {
		logger->warn("No valid data fetched");
		return false;
	}

	//TODO: actions or labels?
	trainingProxy.updateNet({inputs, {}, labels, actions, rewards});

	trainEpoch ++;
	logger->warn("Net updated {} epoch", trainEpoch);

	if (trainEpoch >= rltest::RlSetting::UpdateThreshold) {
		logger->warn("To update network of client proxy");

		trainEpoch = 0;
		return true;
	}

	logger->warn("Update network continue {} < {}", trainEpoch, rltest::RlSetting::UpdateThreshold);
	return false;
}

//TODO: Make dataQ atomic
template <class NetType, class OptType, class OptOptionType>
void RlTrainWorker<NetType, OptType, OptOptionType>::trainingWork() {
//	for (int i = 0; i < netProxies.size(); i ++) {
//		auto data = netProxies[i]->getStates();
//		if (data.size() == 0) {
//			logger->warn("No more data generated by net {}", i);
//			continue;
//		}
//
//		dataQ.push(std::move(data));
//	}
	while (!DataStoreQ::GetDataQ().isEmpty()) {
		auto data = DataStoreQ::GetDataQ().pop();
		if (data[0].size() == 0) {
			logger->warn("An invalid data ");
			continue;
		}

		dataQ.push(std::move(data));
	}

	bool updated = trainingNet();
	if (!updated) {
		return;
	}

	//All proxies waiting for updating
	std::vector<std::shared_ptr<std::promise<bool>>> promises;

	for (int i = 0; i < netProxies.size(); i ++) {
		auto promiseObj = std::make_shared<std::promise<bool>>();
		promises.push_back(promiseObj);
		netProxies[i]->setUpdating(promiseObj);

		logger->warn("Proxy {} set updating", netProxies[i]->getName());
	}
	for (int i = 0; i < promises.size(); i ++) {
		logger->warn("Waiting for promise {}", i);
		auto futureObj = promises[i]->get_future();
		futureObj.get();
		logger->warn("Got promise {}", i);
	}

	//clear dirty data
	while (!DataStoreQ::GetDataQ().isEmpty()) {
		DataStoreQ::GetDataQ().pop();
	}
	logger->warn("Cleared all data produced by previous network ");

	auto newNet = trainingProxy.getTrainingNet();
	//update all proxies
	for (int i = 0; i < netProxies.size(); i ++) {
		netProxies[i]->setRunning(newNet);
		logger->warn("Proxy {} updated network ", netProxies[i]->getName());
	}

	trainingProxy.switchNet();
}

template <class NetType, class OptType, class OptOptionType>
void RlTrainWorker<NetType, OptType, OptOptionType>::start() {
	if (rltest::RlSetting::ProxyNum > rltest::RlSetting::Names.size()) {
		logger->error("Client names not match proxy number: {} != {}", rltest::RlSetting::ProxyNum, rltest::RlSetting::Names.size());
		return;
	}

	logger->error("------------------------------------> Try to start worker");
//	std::cout << "----------------------------------------------> Try to start worker " << std::endl;
	RandomPolicy policy(1.0);

	const int proxyNum = rltest::RlSetting::ProxyNum;
//	std::vector<BaseState> innerStates(proxyNum, BaseState(72, 5));

//	NetProxy<GRUStepNet> netProxy(std::shared_ptr<GRUStepNet>(new GRUStepNet()), innState, policy);
	//TODO: Clarify move/forward/copy constructor/move constructor
	auto net = trainingProxy.getWorkingNet();
	for (int i = 0; i < proxyNum; i ++) {
//		netProxies.push_back(std::forward<NetProxy<NetType>>(NetProxy<NetType>(rltest::RlSetting::Names[i], net, innerStates[i], policy)));
//		netProxies.push_back(std::move<NetProxy<NetType>>(NetProxy<NetType>(rltest::RlSetting::Names[i], net, innerStates[i], policy)));
//		netProxies.push_back({rltest::RlSetting::Names[i], net, innerStates[i], policy});
//		netProxies.push_back(std::shared_ptr<NetProxy<NetType>>(new NetProxy<NetType>(rltest::RlSetting::Names[i], net, innerStates[i], policy)));
		netProxies.push_back(std::shared_ptr<NetProxy<NetType>>(new NetProxy<NetType>(rltest::RlSetting::Names[i], net, {72, 5}, policy)));
	}

//	std::vector<boost::asio::io_context> ios(rltest::RlSetting::ProxyNum);
	boost::asio::io_context io;

//	auto pointer = asiotenhoufsm<GRUStepNet>::Create(io, netProxy, "NoName");
	std::vector<boost::shared_ptr<asiotenhoufsm<NetType>>> clientPointers;
	for (int i = 0; i< proxyNum; i ++) {
//		clientPointers.push_back(asiotenhoufsm<NetType>::Create(ios[i], netProxies[i],
//				rltest::RlSetting::ServerIp, rltest::RlSetting::ServerPort, rltest::RlSetting::Names[i]));
		clientPointers.push_back(asiotenhoufsm<NetType>::Create(io, netProxies[i],
				rltest::RlSetting::ServerIp, rltest::RlSetting::ServerPort, rltest::RlSetting::Names[i]));

	}

	for (int i = 0; i < proxyNum; i ++) {
		clientPointers[i]->start();
	}
//	std::thread ioThread(
//			static_cast<std::size_t (boost::asio::io_context::*)()>(&boost::asio::io_context::run), &io);

	std::vector<std::unique_ptr<std::thread>> ioThreads;
	for (int i = 0; i < 2; i ++) {
		ioThreads.push_back(std::make_unique<std::thread>(
				static_cast<std::size_t (boost::asio::io_context::*)()>(&boost::asio::io_context::run), &io));
	}

	logger->warn("----------------------------------------> All asio fsm started ");

	while (true) {
		trainingWork();
		sleep(100);
	}
}

#endif /* INCLUDE_RLTEST_TRAINWORKER_HPP_ */
