/*
 * lunartest.cpp
 *
 *  Created on: Jun 18, 2020
 *      Author: zf
 */


#include <string.h>
#include <string>
#include <fstream>
#include <iostream>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <torch/torch.h>

#include <cpprl/cpprl.h>

#include <matplotlibcpp.h>

#include "gymtest/communicator.h"
#include "gymtest/requests.h"
#include "gymtest/MlpBase.h"
#include "gymtest/gympolicy.h"
#include "gymtest/meanstd.h"

using Tensor = torch::Tensor;
using string = std::string;
using std::cout;
using std::endl;


//const float actorLossCoef = 1.0;
const int batchSize = 40;
const float clipParam = 0.2;
const float discountFactor = 0.99;
const float entropyCoef = 1e-3;
const float adaLr = 1e-2;
const float rmsLr = 1e-3;
const float rewardClip = 100;
const float valueLossCoef = 0.5;
const float tau = 0.9;

const int maxFrames = 4 * 10e4;
const int epoch = 3;
const int miniBatch = 20;

const int rewardAveWinSize = 10;
const int logInterval = 10;

const string envName = "LunarLander-v2";
const int numEnvs = 4;
const float renderRewardThreshold = 160;

const int hiddenSize = 64;
const bool recurrent = false;
//const bool use_cuda = false;
const int PlotCap = 1024;
std::vector<float> rewardData(PlotCap, 0);
std::vector<float> rewardSmooth(PlotCap, 0);
int plotIndex = 0;

static void plotReward(float reward) {
	plotIndex ++;
	if (plotIndex >= PlotCap) {
		for (int i = 0; i < PlotCap / 2; i ++) {
			rewardData[i] = (rewardData[i * 2] + rewardData[i * 2 + 1]) / 2;
			rewardSmooth[i] = (rewardSmooth[i * 2] + rewardSmooth[i * 2 + 1]) / 2;
		}
		plotIndex = (PlotCap / 2);
	}
	rewardData[plotIndex] = reward;
	rewardSmooth[plotIndex] = rewardSmooth[plotIndex - 1] * 0.95 + reward * 0.05;

	matplotlibcpp::clf();
	matplotlibcpp::grid(true);
	matplotlibcpp::plot(std::vector<float>(&rewardData[0], &rewardData[plotIndex]));
	matplotlibcpp::plot(std::vector<float>(&rewardSmooth[0], &rewardSmooth[plotIndex]));
	matplotlibcpp::pause(0.01);
}

struct StoreDataType {
	std::vector<Tensor> obsv;
	std::vector<Tensor> actionOutput;
	std::vector<Tensor> action;
	std::vector<Tensor> value;
	std::vector<Tensor> reward;
	std::vector<Tensor> dones;

	void clear() {
		obsv.clear();
		actionOutput.clear();
		action.clear();
		value.clear();
		reward.clear();
		dones.clear();
	}
};

template<typename T>
static std::vector<T> flattenVector(std::vector<T> const &input) {
	return input;
}

template <typename T>
static std::vector<T> flattenVector(std::vector<std::vector<T>> const &input)
{
    std::vector<T> output;

    for (auto const &element : input)
    {
        auto sub_vector = flattenVector(element);

        //An alternative to push_back
        output.reserve(output.size() + sub_vector.size());
        output.insert(output.end(), sub_vector.cbegin(), sub_vector.cend());
    }

    return output;
}

static std::tuple<Tensor, cpprl::ActionSpace, cpprl::ActionSpace>
reset(Communicator& comm, const int numEnvs) {
    spdlog::info("Creating environment");
    auto makeParam = std::make_shared<MakeParam>();
    makeParam->env_name = envName;
    makeParam->num_envs = numEnvs;
    Request<MakeParam> makeReq("make", makeParam);
    comm.send_request(makeReq);
    spdlog::info("Response: {}", comm.get_response<MakeResponse>()->result);

    Request<InfoParam> infoReq("info", std::make_shared<InfoParam>());
    comm.send_request(infoReq);
    auto envInfo = comm.get_response<InfoResponse>();
    spdlog::info("Action space: {} - [{}]", envInfo->action_space_type, envInfo->action_space_shape);
    spdlog::info("Observation space: {} - [{}]", envInfo->observation_space_type, envInfo->observation_space_shape);

    spdlog::info("Resetting env");
    auto resetParam = std::make_shared<ResetParam>();
    Request<ResetParam> resetReq("reset", resetParam);
    comm.send_request(resetReq);

    auto obsvShape = envInfo->observation_space_shape;
    obsvShape.insert(obsvShape.begin(), numEnvs);
    std::vector<float> obsvVec = flattenVector(comm.get_response<MlpResetResponse>()->observation);
    Tensor obsv = torch::from_blob(obsvVec.data(), obsvShape);
    spdlog::info("Get reset observation: {}", obsv.sizes());

//    cpprl::ActionSpace actionSpace{envInfo->action_space_type, envInfo->action_space_shape};
//    std::unique_ptr<cpprl::ActionSpace> actionSpace(new cpprl::ActionSpace{envInfo->action_space_type, envInfo->action_space_shape});
    cpprl::ActionSpace actionSpace{envInfo->action_space_type, envInfo->action_space_shape};
    cpprl::ActionSpace obsvSpace{envInfo->observation_space_type, obsvShape};
    return std::make_tuple(obsv, actionSpace, obsvSpace);
}

static std::vector<Tensor>
	getStepResult(Communicator& comm, std::vector<std::vector<float>> actions, const bool render,
			const std::vector<int64_t>& obsShape) {
	auto stepParam = std::make_shared<StepParam>();
    stepParam->actions = actions;
    stepParam->render = render;
    Request<StepParam> stepReq("step", stepParam);
    comm.send_request(stepReq);

    auto stepResult = comm.get_response<MlpStepResponse>();
    auto obsvVec = flattenVector(stepResult->observation);
//    Tensor obsvTensor = torch::from_blob(obsvVec.data(), obsShape);
    Tensor obsvTensor = torch::zeros(obsShape);
    auto dataPtr = obsvTensor.data<float>();
    for (int i = 0; i < obsvVec.size(); i ++) {
    	dataPtr[i] = obsvVec[i];
    }
//    spdlog::info("Get obsv vector: {}", obsvVec);
//    spdlog::info("Get obsvShape: {}", obsShape);
//    std::cout << "orig obsv " << std::endl << obsvTensor << std::endl;

    auto rawRewardVec = flattenVector(stepResult->real_reward);
//    spdlog::info("rawRewardVec: {}", rawRewardVec);
//    spdlog::info("rawRewards");
//    for (auto rawReward: rawRewardVec) {
//    	spdlog::info("{}", rawReward);
//    }
    Tensor rewardTensor = torch::zeros({numEnvs});
    dataPtr = rewardTensor.data<float>();
    for (int i = 0; i < rewardTensor.numel(); i ++) {
    	dataPtr[i] = rawRewardVec[i];
    }
//    std::cout << "rewardTensor: " << endl << rewardTensor << std::endl;
    //TODO: clamp

    auto rawDone = stepResult->done;
    std::vector<int64_t>  doneSizes = {	(int)rawDone.size(), (int)rawDone[0].size()};
    auto doneBoolVec = flattenVector(stepResult->done);
    Tensor doneTensor = torch::zeros(doneSizes);
    dataPtr = doneTensor.data_ptr<float>();
    for (int i = 0; i < doneBoolVec.size(); i ++) {
    	if (doneBoolVec[i]) {
    		dataPtr[i] = 1;
    	}
    }

    return {obsvTensor, rewardTensor, doneTensor};
}

//TODO: Mask
static Tensor getReturns(std::vector<float> nextValues, Tensor rewards, Tensor dones) {
	Tensor returns = torch::zeros(rewards.sizes());
	//TODO: How if internal memory not contiguous?
//	spdlog::info("getReturns");
//	spdlog::info("nextValues: {}", nextValues);
//	cout << "rewards " << endl << rewards << endl;

	auto returnData = returns.accessor<float, 3>();
	auto rewardData = rewards.accessor<float, 3>();
	auto doneData = dones.accessor<float, 3>();
	const int step = rewards.size(0);
	const int batch = rewards.size(1);

	//TODO
	for (int i = 0; i < batch; i ++) {
		returnData[step - 1][i][0] = nextValues[i] * discountFactor + rewardData[step - 1][i][0];
	}

	for (int i = step - 2; i >= 0; i --) {
		for (int j = 0; j < batch; j ++) {
			if (doneData[i][j][0]) {
				returnData[i][j][0] = rewardData[i][j][0];
			} else {
				returnData[i][j][0] = returnData[i + 1][j][0] * discountFactor + rewardData[i][j][0];
			}
		}
	}

//	cout << "returns: " << endl << returns << endl;

	return returns;
}

static Tensor getReturns(std::vector<float> nextValues, Tensor rewards, Tensor values, Tensor dones) {
	Tensor returns = torch::zeros(rewards.sizes());
	//TODO: How if internal memory not contiguous?
//	spdlog::info("getReturns");
//	spdlog::info("nextValues: {}", nextValues);
//	cout << "rewards " << endl << rewards << endl;

	auto returnData = returns.accessor<float, 3>();
	auto rewardData = rewards.accessor<float, 3>();
	auto valueData = values.accessor<float, 3>();
	auto doneData = dones.accessor<float, 3>();
	const int step = rewards.size(0);
	const int batch = rewards.size(1);
	std::vector<float> gae(batch, 0);

	//TODO
	for (int i = 0; i < batch; i ++) {
		auto delta = rewardData[step - 1][i][0] + discountFactor * nextValues[i] - valueData[step - 1][i][0];
		gae[i] = delta + discountFactor * tau * gae[i];
		returnData[step - 1][i][0] = gae[i] + valueData[step - 1][i][0];
	}

	for (int i = step - 2; i >= 0; i --) {
		for (int j = 0; j < batch; j ++) {
			float delta = 0.0;
			if (doneData[i + 1][j][0]) {
				delta = rewardData[i][j][0] - valueData[i][j][0];
				gae[j] = delta;
			} else {
				delta = rewardData[i][j][0] + discountFactor * valueData[i + 1][j][0] - valueData[i][j][0];
				gae[j] = delta + discountFactor * tau * gae[j];
			}
			returnData[i][j][0] = gae[j] + valueData[i][j][0];
		}
	}

	return returns;
}

//template <typename OptimzerType>
//static void updateNet(MlpNet& net, OptimzerType& optimizer, StoreDataType& storage, std::vector<float> nextValues) {
////	net.train(true);
//
//	//obsv: step, batch, others
//	Tensor dummy;
//	Tensor obsv = torch::stack(storage.obsv, 0);
//	auto netOutput = net.forward(obsv, dummy, dummy);
//	Tensor values = netOutput[0];
//	Tensor predValues = torch::stack(storage.value, 0);
//
//	Tensor entropyActBase = torch::stack(storage.actionOutput, 0);
//	Tensor actProb = torch::softmax(entropyActBase, -1);
//	Tensor logActProb = torch::log_softmax(entropyActBase, -1);
//	//TODO: Which mean?
//	Tensor entropy = - (actProb * logActProb).mean();
//
//	Tensor rewards = torch::stack(storage.reward, 0).view(values.sizes());
//	Tensor dones = torch::stack(storage.dones, 0).view(values.sizes());
//	Tensor returns = getReturns(nextValues, rewards, dones);
////	Tensor returns = getReturns(nextValues, rewards, predValues, dones);
//	Tensor adv = returns.sub(values);
//	Tensor valueLoss = 0.5 * adv.pow(2).mean();
//
//	Tensor actions = torch::stack(storage.action, 0);
//	Tensor actPi = logActProb.gather(-1, actions);
//	Tensor policyLoss = - (actPi.detach() * adv).mean();
//
//	auto loss = valueLoss * valueLossCoef + policyLoss; // - entropy * entropyCoef;
//	spdlog::info("valueLoss: {}", valueLoss.item<float>());
//	spdlog::info("actionLoss: {}", policyLoss.item<float>());
//	spdlog::info("entropy: {}", entropy.item<float>());
//
//	optimizer.zero_grad();
//	loss.backward();
//	optimizer.step();
//}


template <typename OptimzerType>
static void updateNet(MlpNet& net, OptimzerType& optimizer, StoreDataType& storage, std::vector<float> nextValues) {
	net.train(true);

	//obsv: step, batch, others
	Tensor dummy;
	Tensor obsv = torch::stack(storage.obsv, 0);
	auto netOutput = net.forward(obsv, dummy, dummy);

	Tensor values = netOutput[0];
	Tensor rewards = torch::stack(storage.reward, 0).view(values.sizes());
	Tensor dones = torch::stack(storage.dones, 0).view(values.sizes());
	Tensor returns = getReturns(nextValues, rewards, dones);
//	std::cout << "values: " << values.sizes() << endl;
//	cout << "returns: " << returns << endl << "----------------------------------> " << endl;
	Tensor adv = returns.sub(values);
//	cout << "advs: " << adv << endl << "----------------------------------> " << endl;
	Tensor valueLoss = adv.pow(2).mean();

	Tensor actsBase = netOutput[1];
	Tensor actsProb = torch::softmax(actsBase, -1);
	actsProb = actsProb.clamp(1.21e-7, 1.0f - 1.21e-7);
	Tensor logActsProb = torch::log_softmax(actsBase, -1);
	Tensor actions = torch::stack(storage.action, 0);
	Tensor actionPi = logActsProb.gather(-1, actions);
	auto actionLoss = - (adv.detach() * actionPi).mean(); //detach: adv is constant in backward

	auto entropy = - (actsProb * logActsProb).mean();
//	spdlog::info("entropy: {}", entropy.item<float>());
//	spdlog::info("actionLoss: {}", actionLoss.item<float>());
//	spdlog::info("valueLoss: {}", valueLoss.item<float>());

	auto loss = valueLoss * valueLossCoef + actionLoss - entropy * entropyCoef;


	optimizer.zero_grad();
	loss.backward();
	optimizer.step();
}

//static void updateNet(MlpNet& net, torch::optim::Adagrad& optimizer, StoreDataType& storage) {
////	spdlog::info("To updateNet");
//	//step, batch, others
//	net.train();
//
//	Tensor obsv = torch::stack(storage.obsv, 0);
//	Tensor value = torch::stack(storage.value, 0);
//	Tensor reward = torch::stack(storage.reward, 0).view(value.sizes());
//	Tensor actionOutput = torch::stack(storage.actionOutput, 0);
//	Tensor action = torch::stack(storage.action, 0);
////	cout << "Get sizes: " << endl;
////	cout << "obsv: " << obsv.sizes() << endl;
////	cout << "value: " << value << endl;
////	cout << "reward: " << reward << endl;
////	cout << "actionOutput: " << actionOutput.sizes() << endl;
////	cout << "action: " << action.sizes() << endl;
//	//TODO: clamp, rewards reverse step calculation
//
//
//	Tensor adv = reward.sub(value);
////	cout << "adv: " << adv << endl;
//	auto logActions = torch::log_softmax(actionOutput, -1);
////	cout << "logActions: " << logActions << endl;
//	auto probActions = torch::softmax(actionOutput, 0);
////	cout << "probActions: " << probActions << endl;
//
//	auto valueLoss = adv.square().mean();
////	cout << "valueLoss: " << valueLoss << endl;
//	auto actionPi = logActions.gather(-1, action);
//	auto actionLoss = -(adv.detach() * actionPi).mean();
////	cout << "actionLoss: " << actionLoss << endl;
//
//	probActions = (probActions) / probActions.sum();
////	cout << "probActions: " << probActions << endl;
//	auto logProbActions = torch::log(probActions);
////	cout << "logProbActions: " << logProbActions << endl;
//	auto entropyLoss = (logProbActions * probActions).sum();
////	cout << "entropyLoss: " << entropyLoss << endl;
//
//	auto loss = valueLoss * valueLossCoef + actionLoss - entropyLoss * entropyCoef;
//
//	optimizer.zero_grad();
//	loss.backward();
//	optimizer.step();
//}

static std::vector<Tensor> createTensors() {
	auto t0 = torch::rand({2, 2});
	auto t1 = torch::rand({3, 3});

	std::cout << "t0" << endl << t0 << endl;
	cout << "t1" << endl << t1 << endl;

	return {t0, t1};
}

static void testVecTensor() {
	auto rc = createTensors();
	cout << "Get t0: " << endl << rc[0] << endl;
	cout << "Get t1: " << endl << rc[1] << endl;
}

static void testLunar() {
    spdlog::set_level(spdlog::level::debug);
    spdlog::set_pattern("%^[%T %7l] %v%$");

    torch::manual_seed(0);
    torch::Device device = torch::kCPU;

    spdlog::info("Connecting to gym server");
    Communicator communicator("tcp://127.0.0.1:10201");

    auto resetResult = reset(communicator, numEnvs);
    auto obsv = std::get<0>(resetResult);
    auto actionSpace = std::get<1>(resetResult);
    auto obsvSpace = std::get<2>(resetResult);
    spdlog::info("Get reset result");

    const int inputSize = obsv.size(1);
    const int outputSize = actionSpace.shape[actionSpace.shape.size() - 1];
    MlpNet net(inputSize, outputSize);
    net.to(device);
    spdlog::info("Net created ");
//	torch::optim::Adagrad optimizer(net.parameters(), torch::optim::AdagradOptions(adaLr));
	torch::optim::RMSprop optimizer(net.parameters(), torch::optim::RMSpropOptions(rmsLr).eps(1e-8).alpha(0.99));


    GymPolicy policy;

    StoreDataType storage;

    Tensor dummyState;
    Tensor dummyMask;
    Tensor returns = torch::zeros({numEnvs});
    RunningMeanStd returnsRms(1);

    std::vector<float> runningReward(numEnvs, 0);
    const int rewardStoreCap = 40;
    std::vector<float> rewardStore(rewardStoreCap, 0);
    int episode = 0;


    bool render = false;

    const int numUpdates = maxFrames / (batchSize * numEnvs);
    for (int update = 0; update < numUpdates; update ++) {
    	for (int step = 0; step < batchSize; step ++) {
//    		net.eval();
    		auto output = net.forward(obsv, dummyState, dummyMask);
    		//output: linear value, linear act, hiddenState
//    		spdlog::info("Forward step");
//    		auto actorOutput = output[2];
    		auto actsBaseOutput = output[1];

    		Tensor actionTensor;
    		{
    			torch::NoGradGuard policyNoGrad;
    			Tensor actProb = torch::softmax(actsBaseOutput, -1);
//    			cout << "actProb: " << endl << actProb << endl;
    			Tensor clampProb = actProb.clamp(1.21e-7, 1.0f - 1.21e-7);
//    			actionTensor = policy.getAct(clampProb);
    			actionTensor = clampProb.multinomial(1);
    		}

    		auto valueTensor = output[0];
//    		cout << "Get actions: " << actionTensor << endl;
    		long *actionValues = actionTensor.data_ptr<long>();
//    		spdlog::info("Get action values");
    		std::vector<std::vector<float>> actions(numEnvs);
    		for (int i = 0; i < numEnvs; i ++) {
    			actions[i] = {actionValues[i]};
    		}

    		std::vector<Tensor> stepResult = getStepResult(communicator, actions, render, obsvSpace.shape);
    		auto rewards = stepResult[1];
//    		cout << "Get rewards: " << endl << rewards << endl;
    		obsv = stepResult[0];
//    		cout << "next observe: " << endl << obsv << endl;
    		auto doneTensor = stepResult[2];
//    		cout << "done " << endl << doneTensor << endl;
    		auto doneData = doneTensor.data<float>();

    		auto realRewards = rewards.data<float>();
			for (int k = 0; k < numEnvs; k ++) {
				runningReward[k] += realRewards[k];

    			if (doneData[k]) {
//    				cout << "---------------------------------> " << doneData[k] << endl;
    				rewardStore[episode % rewardStoreCap] = runningReward[k];
    				episode ++;

    				auto returnData = returns.data<float>();
    				returnData[k] = 0;
    				runningReward[k] = 0;
    			}
    		}


    		//TODO: actionTensor or actorOut?
//    		spdlog::info("rewards before clamp");
//    		cout << rewards << endl;
    		returns = returns * discountFactor + rewards;
    		returnsRms->update(returns);
    		rewards = torch::clamp(rewards / torch::sqrt(returnsRms->get_variance() + 1e-8), -rewardClip, rewardClip);
//    		rewards = torch::clamp(rewards, -1, 1);
//    		spdlog::info("rewards after clamp");
//    		cout << rewards << endl;

    		storage.obsv.push_back(obsv);
    		storage.actionOutput.push_back(actsBaseOutput); //linear output, as we don't want to calculate log_softmax by log(softmax)
    		storage.action.push_back(actionTensor);
    		storage.reward.push_back(rewards);
    		storage.value.push_back(valueTensor); //linear output
    		storage.dones.push_back(doneTensor);

    		if (step % logInterval == 0) {
    			float aveReward = std::accumulate(rewardStore.begin(), rewardStore.end(), 0.0f);
    			aveReward /= (episode < rewardStoreCap ? episode : rewardStoreCap);
    			spdlog::info("AccuAveReward({} / {}): {}", update, numUpdates, aveReward);


//        		float meanReward = rewards.mean().item<float>();
//        		spdlog::info("udpate({}) mean : {}", update, meanReward);
        		plotReward(aveReward);

        		std::vector<float> nextValues;
        		{
        			torch::NoGradGuard nogradGuard;
        			auto valueReturn = net.forward(obsv, dummyState, dummyMask);
        			float* valueDatas = valueReturn[0].data<float>();
        			for (int k = 0; k < valueReturn[0].numel(); k ++) {
        				nextValues.push_back(valueDatas[k]);
        			}
        		}
    			updateNet(net, optimizer, storage, nextValues);
    			storage.clear();
    		}

    	}
    }

}

static void testMean() {
	std::vector<float> data = {
			-0.860984, -0.375366, 0.0789035, -0.490111,
			-0.892326, -0.424383, 0.0406885, -0.527926
	};

	float accMean = std::accumulate(data.begin(), data.end(), 0.0f);
	spdlog::info("accMean total: {}", accMean);
	accMean /= data.size();
	spdlog::info("accMean: {}", accMean);

	float sum = 0;
	for (auto d: data) {
		sum += d;
	}
	spdlog::info("sumMean total: {}", sum);
	sum /= data.size();
	spdlog::info("sumMean: {}", sum);

	Tensor tensor = torch::from_blob(data.data(), {2, 4});
	Tensor mean = tensor.mean();
	cout << mean.sizes() << endl;
	cout << mean.item<float>() << endl;
}

static void testSlice() {
	std::vector<float> data = {
			0,1,2,3,4,5,6,7,8,9,10,11,12
	};
	Tensor tensor = torch::from_blob(data.data(), {3,4});
	cout << "tensor " << endl << tensor << endl;

	Tensor slice = tensor.slice(0, 0, -1);
	cout << "slice " << endl << slice << endl;
}

int main(int argc, char** argv) {
//	testVecTensor();
//	testLunar();
//	testMean();
	testSlice();
}





