/*
 * lunartest2.cpp
 *
 *  Created on: Jul 2, 2020
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
#include "gymtest/gympolicy.h"
#include "gymtest/meanstd.h"
#include "gymtest/gymnetutils.h"
#include <stdexcept>


using Tensor = torch::Tensor;
using string = std::string;
using std::cout;
using std::endl;
using TensorList = torch::TensorList;
using std::vector;




struct Lunar2Net: torch::nn::Module {
private:
	const unsigned int numInputs;
	const unsigned int numActOutput;
	torch::nn::Linear ah;
	torch::nn::Linear vh;
	torch::nn::Linear ah1;
	torch::nn::Linear vh1;
	torch::nn::Linear ao;
	torch::nn::Linear vo;


	Tensor getReturns(std::vector<float> nextValues, Tensor rewards, Tensor values, Tensor dones) {
		static const float discountFactor = 0.99;
		static const float tau = 0.9;

		Tensor returns = torch::zeros(rewards.sizes());
		//TODO: How if internal memory not contiguous?
	//	spdlog::info("getReturns");
	//	spdlog::info("nextValues: {}", nextValues);
	//	cout << "rewards " << endl << rewards << endl;

		auto returnData = returns.accessor<float, 3>();
		auto rewardData = rewards.accessor<float, 3>();
		auto doneData = dones.accessor<float, 3>();
		auto valueData = values.accessor<float, 3>();
		const int step = rewards.size(0);
		const int batch = rewards.size(1);
		vector<float> gae(batch, 0);

		for (int i = 0; i < batch; i ++){
			if (doneData[step - 1][i][0]) {
				auto delta = rewardData[step - 1][i][0] - valueData[step - 1][i][0];
				gae[i] = delta; //next step
				returnData[step - 1][i][0] = gae[i] + valueData[step - 1][i][0];
			} else {
				auto delta = rewardData[step - 1][i][0] + discountFactor * nextValues[i] - valueData[step - 1][i][0];
				gae[i] = delta + discountFactor * tau * gae[i];
				returnData[step - 1][i][0] = gae[i] + valueData[step - 1][i][0];
			}
		}

		for (int i = step - 2; i >= 0; i --) {
			for (int j = 0; j < batch; j ++) {
				if (doneData[i][j][0]) {
					auto delta = rewardData[i][j][0] - valueData[i][j][0];
					gae[j] = delta;
					returnData[i][j][0] = gae[j] + valueData[i][j][0];
				} else {
					auto delta = rewardData[i][j][0] + discountFactor * valueData[i + 1][j][0] - valueData[i][j][0];
					gae[j] = delta + discountFactor * tau * gae[j];
					returnData[i][j][0] = gae[j] + valueData[i][j][0];
				}
			}
		}

	//	cout << "returns: " << endl << returns << endl;

		return returns;
	}

public:
	Lunar2Net(unsigned iNumInputs, unsigned int iNumActOutput, bool recurrent, unsigned int hiddenSize)
		: numInputs(iNumInputs),
		  numActOutput(iNumActOutput),
		  ah(torch::nn::Linear(iNumInputs, hiddenSize)),
		  vh(torch::nn::Linear(iNumInputs, hiddenSize)),
		  ah1(torch::nn::Linear(hiddenSize, hiddenSize)),
		  vh1(torch::nn::Linear(hiddenSize, hiddenSize)),
		  ao(torch::nn::Linear(hiddenSize, numActOutput)),
		  vo(torch::nn::Linear(hiddenSize, 1))
	{
		recurrent = false;

		register_module("ah", ah);
		register_module("vh", vh);
		register_module("ah1", ah1);
		register_module("vh1", vh1);
		register_module("ao", ao);
		register_module("vo", vo);

		init_weights(ah->named_parameters(), sqrt(2.0), 0);
		init_weights(vh->named_parameters(), sqrt(2.0), 0);
		init_weights(ah1->named_parameters(), sqrt(2.0), 0);
		init_weights(vh1->named_parameters(), sqrt(2.0), 0);
		init_weights(ao->named_parameters(), sqrt(2.0), 0);
		init_weights(vo->named_parameters(), sqrt(2.0), 0);
	}

	vector<Tensor> forward(Tensor inputs) {
		Tensor values = vh->forward(inputs);
		values = torch::leaky_relu(values);
		values = vh1->forward(values);
		values = torch::leaky_relu(values);
		values = vo->forward(values);

		Tensor actions = ah->forward(inputs);
		actions = torch::leaky_relu(actions);
		actions = ah1->forward(actions);
		actions = torch::leaky_relu(actions);
		actions = ao->forward(actions);

		return {values, actions};
	}

	Tensor getActions(Tensor inputs) {
		this->eval();
		vector<Tensor> output = forward(inputs);
		Tensor actOutput = output[1];
		Tensor actProb = torch::softmax(actOutput, -1);
		actProb = actProb.clamp(1.21e-7, 1.0f - 1.21e-7);

		Tensor actions = actProb.multinomial(1, true);
		return actions;
	}

	Tensor getLoss(Tensor inputs, Tensor actions, Tensor actReturn) {
		this->train();
		vector<Tensor> output = forward(inputs);
		Tensor valueOutput = output[0];
		Tensor actionOutput = output[1];

		Tensor adv = actReturn - valueOutput;
		Tensor valueLoss = 0.5 * adv.pow(2).mean();

//		cout << "inputs " << endl << inputs << endl;
//		cout << "output[1]: " << endl << output[1] << endl;
		Tensor actionLogProbs = torch::log_softmax(output[1], -1);
		Tensor actionProbs = torch::softmax(output[1], -1);
		actionProbs = actionProbs.clamp(1.21e-7, 1.0f - 1.21e-7);
		Tensor entropy = -(actionLogProbs * actionProbs).sum(-1).mean();
//		cout << "actionLogProbs: " << endl << actionLogProbs.sizes() << endl;
//		cout << "entropy: " << (actionLogProbs * actionProbs).sum(-1).sizes() << endl;
//
//		Tensor expActions;
//		if (actionProbs.dim() == 3) {
//			expActions = actionProbs.view({actionProbs.size(0) * actionProbs.size(1), actionProbs.size(2)}).multinomial(1, true);
//		} else {
//			expActions = actionProbs.multinomial(1, true);
//		}
//		expActions = expActions.view(actions.sizes());

		Tensor actPi = actionLogProbs.gather(-1, actions);
//		cout << "actPi: " << endl << actPi << endl;
//		cout << "adv: " << endl << adv.sizes() << endl;
//		cout << "actions: " << endl << actions << endl;
//		cout << "actPid: " << endl << actPi << endl;
		Tensor actionLoss = (-1) * (actPi * adv.detach()).mean();

		cout << "valueLoss: " << valueLoss.item<float>() << endl;
//		cout << "actionProbs: " << endl << actionProbs << endl;
//		cout << "actions: " << endl << actions << endl;
//		cout << "expActions: " << endl << expActions << endl;
//		cout << "actPi: " << endl << actPi.sizes() << endl;
//		cout << "adv: " << endl << adv << endl;
		cout << "actionLoss: " << actionLoss.item<float>() << endl;
		cout << "entropy: " << entropy.item<float>() << endl;
		cout << "-----------------------------------------> " << endl;

		Tensor loss = valueLoss + actionLoss;
//		- entropy * 1e-4;
		return loss;
	}


//	Tensor getReturns(std::vector<float> nextValues, Tensor rewards, Tensor values, Tensor dones) {
	Tensor getLoss(Tensor inputs, Tensor actions, Tensor rewards, Tensor dones, vector<float> nextValues) {
		this->train();
		vector<Tensor> output = forward(inputs);
		Tensor valueOutput = output[0];
		Tensor actionOutput = output[1];


		Tensor actReturn = getReturns(nextValues, rewards, valueOutput, dones);

		Tensor adv = actReturn - valueOutput;
		Tensor valueLoss = 0.5 * adv.pow(2).mean();

		Tensor actionLogProbs = torch::log_softmax(output[1], -1);
		Tensor actionProbs = torch::softmax(output[1], -1);
		actionProbs = actionProbs.clamp(1.21e-7, 1.0f - 1.21e-7);
		Tensor entropy = -(actionLogProbs * actionProbs).sum(-1).mean();
//
//		Tensor expActions;
//		if (actionProbs.dim() == 3) {
//			expActions = actionProbs.view({actionProbs.size(0) * actionProbs.size(1), actionProbs.size(2)}).multinomial(1, true);
//		} else {
//			expActions = actionProbs.multinomial(1, true);
//		}
//		expActions = expActions.view(actions.sizes());

		Tensor actPi = actionLogProbs.gather(-1, actions);
//		cout << "actionLogProbs: " << endl << actionLogProbs << endl;
//		cout << "actions: " << endl << actions << endl;
//		cout << "actPid: " << endl << actPi << endl;
		Tensor actionLoss = (-1) * (actPi * adv.detach()).mean();

		cout << "valueLoss: " << valueLoss.item<float>() << endl;
//		cout << "actionProbs: " << endl << actionProbs << endl;
//		cout << "actions: " << endl << actions << endl;
//		cout << "expActions: " << endl << expActions << endl;
//		cout << "actPi: " << endl << actPi << endl;
//		cout << "adv: " << endl << adv << endl;
		cout << "actionLoss: " << actionLoss.item<float>() << endl;
		cout << "entropy: " << entropy.item<float>() << endl;
		cout << "-----------------------------------------> " << endl;

		Tensor loss = valueLoss + actionLoss
		- entropy * 1e-4;
		return loss;
	}
};



/******************************************* Game Params *******************************************************/
#define GameParams 1
const int numEnvs = 6;
const string envName = "LunarLander-v2";
const float rmsLr = 1e-3;
const float adaLr = 1e-3;

const int batchSize = 40;
const float clipParam = 0.2;
const float discountFactor = 0.99;
const float entropyCoef = 1e-3;
const float rewardClip = 100;
const float valueLossCoef = 0.5;
const float tau = 0.9;

const int maxFrames = 4 * 10e5;
const int epoch = 3;
const int miniBatch = 20;

const int rewardAveWinSize = 10;
const int logInterval = 10;
const int render_reward_threshold = 160;
/****************************************** Game Utils ***********************************************************/
#define GameUtils 1

const int PlotCap = 1024;
std::vector<float> rewardData(PlotCap, 0);
std::vector<float> rewardSmooth(PlotCap, 0);
int plotIndex = -1;
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
		action.clear();
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
    auto dataPtr = obsvTensor.data_ptr<float>();
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
    dataPtr = rewardTensor.data_ptr<float>();
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

static vector<float> getNextValues(Lunar2Net& net, Tensor input, Tensor done) {
//	cout << "input sizes: " << input.sizes() << endl;
	auto inputs = input.split(1, 0);
	float* dones = done.data_ptr<float>();

	vector<float> values(inputs.size(), 0.0f);
	for (int i = 0; i < inputs.size(); i ++) {
		if (!dones[i]) {
			auto output = net.forward(inputs[i]);
//			cout << "inputs[i]: " << inputs[i] << endl;
//			cout << "output[0]: " << output[0] << endl;
//			cout << "output[1]: " << output[1] << endl;
			values[i] = output[0].item<float>();
		}
	}

	return values;
}


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
/********************************************* Training *************************************************************/
#define TRAIN 1
//#define Comma "
//#define printSize(tensorName) { \
//	cout << Comma##tensorName##Comma << tensorName.sizes() << endl; \
//}

template <typename OptimzerType>
static void updateNet(Lunar2Net& net, OptimzerType& optimizer, StoreDataType& storage, std::vector<float> nextValues) {
	//obsv: step, batch, others
	Tensor obsv = torch::stack(storage.obsv, 0);
//	cout << "obsv vec size: " << storage.obsv.size() << endl;
//	cout << "obsv elem size: " << storage.obsv[0].sizes() << endl;
	Tensor dones = torch::stack(storage.dones, 0);
	Tensor rewards = torch::stack(storage.reward, 0).view(dones.sizes());
	Tensor actions = torch::stack(storage.action, 0).view(dones.sizes());
//	cout << "obsv size: " << obsv.sizes() << endl;
//	cout << "rewards " << rewards.sizes() << endl;
//	cout << "actions " << actions.sizes() << endl;
//	cout << "dones " << dones.sizes() << endl;

	Tensor actReturns = getReturns(nextValues, rewards, dones);
//	Tensor actReturns = getReturns(nextValues, rewards, )

//	Tensor getLoss(Tensor inputs, Tensor actions, Tensor rewards, Tensor dones, vector<float> nextValues) {
//	Tensor loss = net.getLoss(obsv, actions, rewards, dones, nextValues);
	Tensor loss = net.getLoss(obsv, actions, actReturns);

	optimizer.zero_grad();
	loss.backward();
	optimizer.step();
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
    Lunar2Net net(inputSize, outputSize, false, 64);
    net.to(device);
    spdlog::info("Net created ");
//	torch::optim::Adagrad optimizer(net.parameters(), torch::optim::AdagradOptions(adaLr));
	torch::optim::RMSprop optimizer(net.parameters(), torch::optim::RMSpropOptions(rmsLr).eps(1e-8).alpha(0.99));
//    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(1e-4));

    StoreDataType storage;

    Tensor returns = torch::zeros({numEnvs});
    Tensor doneTensor = torch::zeros({numEnvs, 1});
    RunningMeanStd returnsRms(1);

    std::vector<float> runningReward(numEnvs, 0);
    const int rewardStoreCap = 40;
    std::vector<float> rewardStore(rewardStoreCap, 0);
    int episode = 0;
    bool render = false;

    const int numUpdates = maxFrames / (batchSize * numEnvs);
     for (int update = 0; update < numUpdates; update ++) {
     	for (int step = 0; step < batchSize; step ++) {
     		storage.obsv.push_back(obsv);
     		storage.dones.push_back(doneTensor);
//     		cout << "obsv: " << obsv << endl;

     		Tensor actionTensor = net.getActions(obsv);
     		auto actionValues = actionTensor.data_ptr<long>();
     		std::vector<std::vector<float>> actions(numEnvs);
     		for (int i = 0; i < numEnvs; i ++) {
     			actions[i] = {actionValues[i]};
     		}

     		std::vector<Tensor> stepResult = getStepResult(communicator, actions, render, obsvSpace.shape);
     		auto rewards = stepResult[1];
 //    		cout << "Get rewards: " << endl << rewards << endl;
     		obsv = stepResult[0];
 //    		cout << "next observe: " << endl << obsv << endl;
     		doneTensor = stepResult[2];
//     		spdlog::info("doneTensor sizes: {}", doneTensor.sizes());
 //    		cout << "done " << endl << doneTensor << endl;
     		auto doneData = doneTensor.data_ptr<float>();

     		auto realRewards = rewards.data_ptr<float>();
 			for (int k = 0; k < numEnvs; k ++) {
 				runningReward[k] += realRewards[k];

     			if (doneData[k]) {
 //    				cout << "---------------------------------> " << doneData[k] << endl;
     				rewardStore[episode % rewardStoreCap] = runningReward[k];
     				episode ++;

     				auto returnData = returns.data_ptr<float>();
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

     		storage.action.push_back(actionTensor);
     		storage.reward.push_back(rewards);
     	}
 		if ((update + 1) % logInterval == 0) {
 			float aveReward = std::accumulate(rewardStore.begin(), rewardStore.end(), 0.0f);
 			aveReward /= (episode < rewardStoreCap ? episode : rewardStoreCap);
 			spdlog::info("AccuAveReward({} / {}): {}", update, numUpdates, aveReward);


     		plotReward(aveReward);


 			vector<float> nextValues = getNextValues(net, obsv, doneTensor);
 			updateNet(net, optimizer, storage, nextValues);
 			storage.clear();

 			render = aveReward >= render_reward_threshold;
 		}
     }

}


/**************************************************** Main *********************************************/
static void testSample() {
	vector<float> data = {
			0.3143,  0.2356,  0.2661,  0.1840,
			  0.2804,  0.3582,  0.1995,  0.1618,
			  0.3010,  0.2840,  0.2491,  0.1658,
			  0.2882,  0.3000,  0.2679,  0.1440,
	};

	Tensor probs = torch::from_blob(data.data(), {4, 4});
	Tensor actions = probs.multinomial(1, false);

	cout << "probs: " << endl << probs << endl;
	cout << "actions: " << endl << actions << endl;
}

#define Main 1
int main(int argc, char** argv) {
//	testSample();
	testLunar();
}
