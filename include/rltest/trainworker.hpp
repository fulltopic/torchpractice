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
#include "utils/teststats.h"

#include "returncal.h"
#include "rltestsetting.h"

template <class NetType, class OptType, class OptOptionType>
class TrainObj {
private:
	int curIndex;
	int lastIndex;
	uint32_t epochNum;
	uint32_t totalSampleNum;
	std::shared_ptr<spdlog::logger> logger;

public:
	enum {Cap = 2,};
	std::vector<std::shared_ptr<NetType>> nets;
	std::vector<std::unique_ptr<OptType>> optimizers;


	explicit TrainObj(): curIndex(0), lastIndex(1), epochNum(0), totalSampleNum(0), logger(Logger::GetLogger()) {
	}

	//TODO: Other constructors
	TrainObj(const TrainObj<NetType, OptType, OptOptionType>& other) = delete;

	template<typename ...NetArgs>
	void createNets(NetArgs&&... args) {
		nets.push_back(std::move(std::shared_ptr<NetType>(new NetType(args...))));
		logger->warn("Create original net");

		for (int i = 1; i < Cap; i ++) {
			auto copy = nets[0]->clone();
			std::shared_ptr<NetType> cpyNet = std::dynamic_pointer_cast<NetType>(copy);
			nets.push_back(cpyNet);

			logger->warn("Cloned double net");
		}
	}

	template<typename ...OptArgs>
	void createOptimizers(std::string optModelPath, OptArgs&&... args) {
		for (int i = 0; i < Cap; i ++) {
			//TODO: move?
//			optimizers.push_back(OptType(nets[i]->parameters(), args...));
//			optimizers.push_back(std::make_unique<OptType>(nets[i]->parameters(), args...));
			auto opt = std::make_unique<OptType>(nets[i]->parameters(), args...);
			if (optModelPath.size() > 0) {
				logger->info("To load optimizer model from {}", optModelPath);
				try {
				torch::serialize::InputArchive inChive;
				inChive.load_from(optModelPath);
				opt->load(inChive);
				logger->info("Loaded optimizer ");
				} catch (std::exception& e) {
					logger->error("Failed to load optimizer model as: {}", e.what());
				}
			}

			optimizers.push_back(std::move(opt));
			logger->warn("Create optimizer {}", i);
		}
	}

	//working network for agents
	std::shared_ptr<NetType> getWorkingNet() {
		logger->warn("Get current working(not training) net {}", lastIndex);
		return nets[lastIndex];
	}

	//network that is being updated
	std::shared_ptr<NetType>& getTrainingNet() {
		return nets[curIndex];
	}


	void saveNetOpt() {
		if (epochNum < rltest::RlSetting::SaveEpochThreshold) {
			return;
		}

		auto saveTime = std::chrono::system_clock::now().time_since_epoch();
		auto saveSecond = std::chrono::duration_cast<std::chrono::seconds>(saveTime).count();
		std::string fileName = NetType::GetName() + "_"
				+ std::to_string(epochNum) + "_"
				+ std::to_string(totalSampleNum) + "_"
				+ std::to_string(saveSecond);
		std::string modelPath = rltest::RlSetting::ModelDir + "/" + fileName + ".pt";
		logger->info("To save model into {}", modelPath);

		epochNum = 0;
		totalSampleNum = 0;

		torch::serialize::OutputArchive output_archive;
		try {
			getTrainingNet()->save(output_archive);
			output_archive.save_to(modelPath);
			logger->info("Model saved into {}", modelPath);
		} catch (std::exception& e) {
			logger->error("Failed to save model as: {}", e.what());
		}

		std::string optTypeName = typeid(OptType).name();
		std::string optFileName = optTypeName + "_"
				+ std::to_string(epochNum) + "_"
				+ std::to_string(totalSampleNum) + "_"
				+ std::to_string(saveSecond);
		std::string optModelPath = rltest::RlSetting::ModelDir + "/" + optFileName + ".pt";
		logger->info("To save opt into {}", optModelPath);

		torch::serialize::OutputArchive optOutputArchive;
		try {
			optimizers[curIndex]->save(optOutputArchive);
			optOutputArchive.save_to(optModelPath);
			logger->info("Optimizer saved");
		} catch(std::exception& e){
			logger->error("Failed to save optimizer model as: {}", e.what());
		}
	}

	void updateNet(std::vector<std::vector<torch::Tensor>> inputs, const int sampleNum) {
		logger->warn("Get loss of nets {}", curIndex);
		torch::Tensor loss = nets[curIndex]->getLoss(inputs);

		logger->warn("Optimizer by optimizer {}", curIndex);
		optimizers[curIndex]->zero_grad();
		loss.backward();
		optimizers[curIndex]->step();

		logger->warn("Worker set Net {} updated", curIndex);

		epochNum ++;
		totalSampleNum += sampleNum;

		saveNetOpt();
	}

	void switchNet() {
		auto copy = nets[curIndex]->clone();
		std::shared_ptr<NetType> cpyNet = std::dynamic_pointer_cast<NetType>(copy);
		nets[lastIndex] = cpyNet;
		logger->warn("Cloned net {} into net {}", curIndex, lastIndex);

		//TODO: Check if it is not rmspro optimizer
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
		logger->warn("Optimizer updated ");

		int tmp = curIndex;
		curIndex = lastIndex;
		lastIndex = tmp;
		logger->warn("Switched. curIndex = {}, lastIndex = {}", curIndex, lastIndex);
	}
};


template <class NetType, class OptType, class OptOptionType>
class RlTrainWorker {
using StoreDataType = std::vector<std::vector<torch::Tensor>>;

private:
	int trainEpoch;

	rltest::ReturnCalculator& calc;

	//TODO: inject innstate and policy
	std::vector<std::shared_ptr<NetProxy<NetType>>> netProxies;
	std::vector<bool> workingStatus;


//	std::queue<StoreDataType> dataQ;
	std::queue<std::unique_ptr<StateDataType>> dataQ; //Buffer, for tmp peak

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
		std::unique_ptr<StateDataType> storeData = std::move(dataQ.front());
		dataQ.pop();
		auto datas = storeData->getData();

		std::vector<torch::Tensor> inputData = datas[InputIndex];
		if (inputData.size() == 0) {
			logger->warn("popped invalid data");
			continue;
		}
		torch::Tensor input = torch::cat(inputData, 0); // {seqLen, 5, 72}
		inputs.push_back(input);

		std::vector<torch::Tensor> actionData = datas[ActionIndex];
		torch::Tensor action = torch::cat(actionData, 0);
		actions.push_back(action);

		std::vector<torch::Tensor> labelData = datas[LabelIndex];
		torch::Tensor label = torch::stack(labelData, 0);
		labels.push_back(label);

		torch::Tensor reward = datas[RewardIndex][0];
		rewards.push_back(reward);
	}
	if (rewards.size() == 0) {
		logger->warn("No valid data fetched");
		return false;
	}

	//TODO: actions or labels?
	trainingProxy.updateNet({inputs, {}, labels, actions, rewards}, inputs.size());

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
		std::unique_ptr<StateDataType> data = DataStoreQ::GetDataQ().pop();
		if (data->trainStates.size() == 0) {
			logger->warn("An invalid data ");
			continue;
		}

		dataQ.push(std::move(data));
	}

	if (!trainingNet()) {
		return;
	}

	//All proxies waiting for updating
	std::vector<std::shared_ptr<std::promise<bool>>> promises;

	for (int i = 0; i < netProxies.size(); i ++) {
			if (workingStatus[i]) {
				auto promiseObj = std::make_shared<std::promise<bool>>();
				promises.push_back(promiseObj);
				netProxies[i]->setUpdating(promiseObj);

				logger->warn("Proxy {} set updating", netProxies[i]->getName());
		} else {
			logger->info("Proxy {} is not working, ignore update", netProxies[i]->getName());
		}
	}
	for (int i = 0; i < promises.size(); i ++) {
		if (workingStatus[i]) {
			logger->warn("Waiting for promise {}", i);
			auto futureObj = promises[i]->get_future();
			workingStatus[i] = futureObj.get();
			logger->warn("Got promise {}", i);
		}
	}

	//clear dirty data
	while (!DataStoreQ::GetDataQ().isEmpty()) {
		DataStoreQ::GetDataQ().pop();
	}
	logger->warn("Cleared all data produced by previous network ");

	auto& newNet = trainingProxy.getTrainingNet();
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

	logger->info("create statistics recorder");
	auto recorder = RlTestStatRecorder::GetRecorder(rltest::RlSetting::StatsDataName);

	logger->error("------------------------------------> Try to start worker");
	RandomPolicy policy(0.95);
	RandomPolicy rnPolicy(0.0);

	const int proxyNum = rltest::RlSetting::ProxyNum;
//	std::vector<BaseState> innerStates(proxyNum, BaseState(72, 5));

	//TODO: Clarify move/forward/copy constructor/move constructor
	auto net = trainingProxy.getWorkingNet();
	for (int i = 0; i < proxyNum; i ++) {
		if (rltest::RlSetting::IsPrivateTest) {
			if ((i % 2) == 0) {
				netProxies.push_back(std::shared_ptr<NetProxy<NetType>>(
					new NetProxy<NetType>(rltest::RlSetting::Names[i], net, {72, 5}, policy)));
				logger->info("proxy {} created with policy", netProxies[i]->getName(), false);
			} else {
				netProxies.push_back(std::shared_ptr<NetProxy<NetType>>(
					new NetProxy<NetType>(rltest::RlSetting::Names[i], net, {72, 5}, rnPolicy)));
				logger->info("proxy {} created with random policy", netProxies[i]->getName(), true);
			}
		} else {
			netProxies.push_back(std::shared_ptr<NetProxy<NetType>>(
					new NetProxy<NetType>(rltest::RlSetting::Names[i], net, {72, 5}, policy)));
			logger->info("proxy {} created with policy", netProxies[i]->getName(), false);
		}
	}

	logger->info("To set proxy working stats");
	workingStatus = std::vector<bool>(netProxies.size(), true);

	logger->info("To prepare stats recorder");
	for (int i = 0; i < proxyNum; i ++){
		netProxies[i]->setStatsRecorder(recorder);
	}
	//TODO: why &record is invalid?
	std::thread recordThread(&RlTestStatRecorder::write2File, recorder);

	logger->info("Create fsm with net ");
	boost::asio::io_context io;

	std::vector<boost::shared_ptr<asiotenhoufsm<NetType>>> clientPointers;
	for (int i = 0; i< proxyNum; i ++) {
		clientPointers.push_back(asiotenhoufsm<NetType>::Create(io, netProxies[i],
				rltest::RlSetting::ServerIp, rltest::RlSetting::ServerPort, rltest::RlSetting::Names[i], rltest::RlSetting::IsPrivateTest));

	}

	logger->info("To start all client fsm");
	for (int i = 0; i < proxyNum; i ++) {
		clientPointers[i]->start();
	}

	logger->info("Launch threads for io service");
	std::vector<std::unique_ptr<std::thread>> ioThreads;
	for (int i = 0; i < rltest::RlSetting::ThreadNum; i ++) {
		ioThreads.push_back(std::make_unique<std::thread>(
				static_cast<std::size_t (boost::asio::io_context::*)()>(&boost::asio::io_context::run), &io));
	}

	logger->warn("----------------------------------------> All asio fsm started ");

	logger->info("Ready to train");
	while (true) {
		trainingWork();
		sleep(100);
	}
}

#endif /* INCLUDE_RLTEST_TRAINWORKER_HPP_ */
