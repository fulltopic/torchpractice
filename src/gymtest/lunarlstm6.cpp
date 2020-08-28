/*
 * lunarlstm3.cpp
 *
 *  Created on: Jul 13, 2020
 *      Author: zf
 */



#include <string.h>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>

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




struct LunarRnnNet3: torch::nn::Module {
private:
	const unsigned int numInputs;
	const unsigned int numActOutput;
	torch::nn::Linear ah;
	torch::nn::Linear vh;
	torch::nn::Linear vh1;
	torch::nn::GRU gru;
	torch::nn::Linear ao;
	torch::nn::Linear vo;

public:
	LunarRnnNet3(unsigned iNumInputs, unsigned int iNumActOutput, unsigned int hiddenSize)
		: numInputs(iNumInputs),
		  numActOutput(iNumActOutput),
		  ah(torch::nn::Linear(hiddenSize, hiddenSize)),
		  vh(torch::nn::Linear(hiddenSize, hiddenSize)),
		  vh1(torch::nn::Linear(hiddenSize, hiddenSize)),
		  gru(torch::nn::GRUOptions(iNumInputs, hiddenSize).num_layers(1)),
		  ao(torch::nn::Linear(hiddenSize, numActOutput)),
		  vo(torch::nn::Linear(hiddenSize, 1))
	{
		register_module("gru", gru);
		register_module("ah", ah);
		register_module("ao", ao);
		register_module("vh", vh);
		register_module("vh1", vh1);
		register_module("vo", vo);

		init_weights(ah->named_parameters(), sqrt(2.0), 0);
		init_weights(vh->named_parameters(), sqrt(2.0), 0);
		init_weights(vh1->named_parameters(), sqrt(2.0), 0);
		init_weights(gru->named_parameters(), 1, 0);
		init_weights(ao->named_parameters(), sqrt(2.0), 0);
		init_weights(vo->named_parameters(), sqrt(2.0), 0);
	}


	vector<Tensor> forward(Tensor inputs) {
//		Tensor hStates = torch::zeros(inputs.sizes());
		auto gruOutputData = gru->forward(inputs);
		Tensor gruOutput = std::get<0>(gruOutputData);

		Tensor valueOutput = vh->forward(gruOutput);
		valueOutput = torch::leaky_relu(valueOutput);
		valueOutput = vh1->forward(valueOutput);
		valueOutput = torch::leaky_relu(valueOutput);
		valueOutput = vo->forward(valueOutput);

		Tensor actionOutput = ah->forward(gruOutput);
		actionOutput = torch::leaky_relu(actionOutput);
		actionOutput = ao->forward(actionOutput);

		return {valueOutput, actionOutput};
	}


	vector<Tensor> forward(Tensor input,  Tensor hState) {
		auto gruOutputData = gru->forward(input, hState);
		Tensor gruOutput = std::get<0>(gruOutputData);
		Tensor newState = std::get<1>(gruOutputData);

//		valueOutput = torch::leaky_relu(valueOutput);
		Tensor valueOutput = vh->forward(gruOutput);
		valueOutput = torch::leaky_relu(valueOutput);
		valueOutput = vh1->forward(valueOutput);
		valueOutput = torch::leaky_relu(valueOutput);
		valueOutput = vo->forward(valueOutput);

		Tensor actionOutput = ah->forward(gruOutput);
		actionOutput = torch::leaky_relu(actionOutput);
		actionOutput = ao->forward(actionOutput);

		return {valueOutput, actionOutput, newState};
	}

	vector<Tensor> getAction(Tensor input, Tensor hState) {
		this->eval();
		vector<Tensor> output = forward(input, hState);
		Tensor actionOutput = output[1].squeeze(1);
		Tensor actProb = torch::softmax(actionOutput, -1);
//		cout << "actionOutput: " << output[1].sizes() << endl;
		actProb = actProb.clamp(1.21e-7, 1.0f - 1.21e-7);

		Tensor action = actProb.multinomial(1, true);
//		return action.item<long>();
		return {action, output[2], output[0]};
	}

	vector<Tensor> preprocess(vector<vector<Tensor>> datas) {
		vector<Tensor> inputs = datas[0];
		vector<Tensor> actions = datas[1];
		vector<Tensor> actReturns = datas[2];
		const int batchSize = inputs.size();

		auto sortFunc = [] (const Tensor& t0, const Tensor& t1) -> bool {
			return t0.size(0) > t1.size(0);
		};

		std::stable_sort(inputs.begin(), inputs.end(), sortFunc);
		std::stable_sort(actions.begin(), actions.end(), sortFunc);
		std::stable_sort(actReturns.begin(), actReturns.end(), sortFunc);

		const int seqLen = inputs[batchSize - 1].size(0);
		vector<Tensor> newInputs(batchSize);
		vector<Tensor> newActions(batchSize);
		vector<Tensor> newReturns(batchSize);

		for (int i = 0; i < batchSize; i ++) {
			newInputs[i] = inputs[i].narrow(0, 0, seqLen);
			newActions[i] = actions[i].narrow(0, 0, seqLen);
			newReturns[i] = actReturns[i].narrow(0, 0, seqLen);
		}

		Tensor input = torch::stack(newInputs, 0);
		Tensor action = torch::stack(newActions, 0);
		Tensor returnTensor = torch::stack(newReturns, 0);

		input = input.transpose(0, 1);
		action = action.transpose(0, 1);
		returnTensor = returnTensor.transpose(0, 1);

		return {input, action, returnTensor};
	}

	Tensor getLoss(vector<Tensor> inputs, vector<Tensor> actions, vector<Tensor> actReturn) {
		//inputs[x]: seq, x
		//vector<> inputs: batch, seq, x
		vector<Tensor> datas = preprocess({inputs, actions, actReturn});
		Tensor inputTensor = datas[0];
		Tensor actionTensor = datas[1];
		Tensor returnTensor = datas[2];

		this->train();

//		cout << "inputTensor: " << endl << inputTensor << endl;
//		cout << "actionTensor: " << endl << actionTensor << endl;
		vector<Tensor> output = forward(inputTensor);
		Tensor valueOutput = output[0];
		Tensor actionOutput = output[1];


		Tensor adv = returnTensor - valueOutput;
//		cout << "adv: " << endl << adv << endl;

		Tensor valueLoss = 0.5 * adv.pow(2).mean();

//		cout << "inputs " << endl << inputs << endl;
//		cout << "output[1]: " << endl << output[1] << endl;
		Tensor actionLogProbs = torch::log_softmax(actionOutput, -1);
		Tensor actionProbs = torch::softmax(actionOutput, -1);
		actionProbs = actionProbs.clamp(1.21e-7, 1.0f - 1.21e-7);
		Tensor entropy = -(actionLogProbs * actionProbs).sum(-1).mean();
		cout << "actionLogProbs: " << endl << actionLogProbs.sizes() << endl;
//		cout << "entropy: " << endl << (actionLogProbs * actionProbs).sum(-1).sizes() << endl;

		Tensor actPi = actionLogProbs.gather(-1, actionTensor);
//		cout << "actPi: " << endl << actPi[0] << endl;
//		cout << "actionLogProbs: " << endl << actionLogProbs[0] << endl;
//		cout << "adv: " << endl << adv.sizes() << endl;
//		cout << "actions: " << endl << actions[0] << endl;
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

		Tensor loss = valueLoss + actionLoss
		- entropy * 1e-3;
		return loss;
	}

};

/********************************************* DataType ****************************************************/
#define DataStorage

//template<typename T>
static std::vector<float> flattenVector(std::vector<float> const &input) {
	return input;
}

//template <typename T>
static std::vector<float> flattenVector(std::vector<std::vector<float>> const &input)
{
    std::vector<float> output;

    for (auto const &element : input)
    {
        auto sub_vector = flattenVector(element);

        //An alternative to push_back
        output.reserve(output.size() + sub_vector.size());
        output.insert(output.end(), sub_vector.cbegin(), sub_vector.cend());
    }

    return output;
}

const float discountFactor = 0.99;
struct StoreDataType {
	vector<vector<float> > obsv;
	std::vector<long> action;
	std::vector<float> reward;
	bool done;
//	float nextValue;

	void addStep(vector<float> ob, long act, float score) {
		obsv.push_back(ob);
		action.push_back(act);
		reward.push_back(score);
	}

	vector<Tensor> getData() {
		const int stepSize = obsv.size();
		vector<float> obsvs = flattenVector(obsv);
		Tensor obsvTensor = torch::zeros({stepSize, (int)obsvs.size() / stepSize});
		float* obsvData = obsvTensor.data_ptr<float>();
		std::copy(obsvs.begin(), obsvs.end(), obsvData);
//		spdlog::info("obsvs 11: {}", obsv[11]);
//		cout << "obsv tensor: " << obsvTensor[11] << endl;


		Tensor actionTensor = torch::zeros({stepSize, 1}, at::kLong);
		long* actionData = actionTensor.data_ptr<long>();
		std::copy(action.begin(), action.end(), actionData);

//		Tensor rewardTensor = torch::zeros({stepSize, 1});
//		float* rewardData = rewardTensor.data_ptr<float>();
//		std::copy(reward.begin(), reward.end(), rewardData);

		Tensor returnTensor = torch::zeros({stepSize, 1});
		float* returnData = returnTensor.data_ptr<float>();
//		returnData[stepSize - 1] = nextValue * discountFactor + reward[stepSize - 1];
		returnData[stepSize - 1] = reward[stepSize - 1];
//		spdlog::info("The ultimate reward: {}", reward[stepSize - 1]);
		for (int i = stepSize - 2; i >= 0; i --) {
			returnData[i] = returnData[i + 1] * discountFactor + reward[i];
		}

		return {obsvTensor, actionTensor, returnTensor};
	}

	void setDone() {
		done = true;
//		nextValue = value;
	}

	bool isDone() {
		return done;
	}

	void reset() {
		done = false;
		obsv.clear();
		action.clear();
		reward.clear();
//		nextValue = 0;
	}
};

struct LunarLstmStorageType {
	const int Cap = 64;
	vector<StoreDataType> datas;
	int lastIndex; //The last index had been written
	int index; //The index to be written

	LunarLstmStorageType(): datas(vector<StoreDataType>(Cap)),
			lastIndex(0),
			index(1)
	{
	}

	void addStep(int index, vector<float> ob, int action, float reward) {
		datas[index].addStep(std::move(ob), action, reward);
	}

	void done(int index) {
		datas[index].setDone();
	}

	bool isDone(int index) {
		return datas[index].isDone();
	}

	int getNextSlot() {
		if (index == lastIndex) {
			spdlog::error("Exceeding capacity {}", Cap);
			return -1;
		}

		int storeIndex = index;
		index = (index + 1) % Cap;

		return storeIndex;
	}

	vector<vector<Tensor>> getData() {
		vector<vector<Tensor>> dataTensor(3);
		int curIndex = (lastIndex + 1) % Cap;
		while (curIndex != index) {
			if (datas[curIndex].reward.size() > 0) {
				vector<Tensor> data = datas[curIndex].getData();
				dataTensor[0].push_back(data[0]);
				dataTensor[1].push_back(data[1]);
				dataTensor[2].push_back(data[2]);
			}
			datas[curIndex].reset();

			curIndex = (curIndex + 1) % Cap;
		}

		lastIndex = (index - 1 + Cap) % Cap;

		return dataTensor;
	}

	//TODO: index
	int getDataSize() {
		return (index + Cap - lastIndex - 1) % Cap;
	}
};

/******************************************* Game Params *******************************************************/
#define GameParams 1
const int numEnvs = 8;
const string envName = "LunarLander-v2";
const float rmsLr = 1e-3;
const float adaLr = 1e-2;

const int batchSize = 40;
const float clipParam = 0.2;
const float entropyCoef = 1e-3;
const float rewardClip = 100;
const float valueLossCoef = 0.5;
const float tau = 0.9;

const int maxFrames = 4 * 10e5;
const int epochNum = 8192 * 2;
const int miniBatch = 8;
const int updateInterval = 64;

const int rewardAveWinSize = 10;
const int logInterval = 10;
const int render_reward_threshold = 160;
/****************************************** Game Utils ***********************************************************/
#define GameUtils 1

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

struct RnnMeanType {
	int count;
	float mean;
	double varSquare;

	RnnMeanType (): count(0), mean(0), varSquare(0) {
	}

	void update(float returnValue) {
		float newMean = (mean * count + returnValue) / (count + 1);
		float deltaMean = newMean - mean;
		mean = newMean;
		varSquare = varSquare + count * std::pow(deltaMean, 2) + std::pow((returnValue - mean), 2);

		count ++;
	}

	float getVar() {
		return varSquare / count;
	}
};


static std::tuple<vector<float>, cpprl::ActionSpace, cpprl::ActionSpace>
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

    cpprl::ActionSpace actionSpace{envInfo->action_space_type, envInfo->action_space_shape};
    cpprl::ActionSpace obsvSpace{envInfo->observation_space_type, obsvShape};
    return std::make_tuple(obsvVec, actionSpace, obsvSpace);
}

static std::tuple<vector<float>, float, bool>
getStepResult(Communicator& comm, int action, const bool render,
		const std::vector<int64_t>& obsShape) {
	auto stepParam = std::make_shared<StepParam>();
	vector<vector<float>> actions(1, vector<float>(1, action));
	stepParam->actions = actions;
	stepParam->render = render;
	Request<StepParam> stepReq("step", stepParam);
	comm.send_request(stepReq);

	auto stepResult = comm.get_response<MlpStepResponse>();
	auto obsvVec = flattenVector(stepResult->observation);

	auto rawRewardVec = flattenVector(stepResult->real_reward);
	float reward = rawRewardVec[0];

    auto rawDone = stepResult->done;
    bool done = rawDone[0][0];

    return {std::move(obsvVec), reward, done};
}


/********************************************* Training *************************************************************/
#define TRAIN 1

template <typename OptimzerType>
static void updateNet(LunarRnnNet3& net, OptimzerType& optimizer, LunarLstmStorageType& storage) {
	//obsv: step, batch, others
	vector<vector<Tensor>> datas = storage.getData();
	vector<Tensor> obsvs = datas[0];
	vector<Tensor> actions = datas[1];
	vector<Tensor> returns = datas[2];


	Tensor loss = net.getLoss(obsvs, actions, returns);

	optimizer.zero_grad();
	loss.backward();
	optimizer.step();
}


static void testLunar() {
    spdlog::set_level(spdlog::level::debug);
    spdlog::set_pattern("%^[%T %7l] %v%$");

    torch::manual_seed(0);
    torch::Device device = torch::kCPU;

    LunarLstmStorageType store;
    vector<int> storeIndex(numEnvs, -1);
    for (int i = 0; i < numEnvs; i ++) {
    	storeIndex[i] = store.getNextSlot();
    }

    vector<Communicator*> comms;
    for (int i = 0; i < numEnvs; i ++) {
    	std::string serverAddr = "tcp://127.0.0.1:1022" + std::to_string(i + 1);
    	spdlog::info("Connecting to gym server {}", serverAddr);
    	comms.push_back(new Communicator(serverAddr));
    }

    vector<vector<float>> obsvs(numEnvs);
//    vector<int> actions(numEnvs);
//    vector<float> rewards(numEnvs);

	auto resetResult = reset(*comms[0], 1);
	obsvs[0] = std::get<0>(resetResult);
	auto actionSpace = std::get<1>(resetResult);
	auto obsvSpace = std::get<2>(resetResult);
	spdlog::info("obsvSpace: {}", obsvSpace.shape);
	const int inputSize = obsvs[0].size();
	const int outputSize = actionSpace.shape[actionSpace.shape.size() - 1];
	spdlog::info("Get reset result {}", 0);
    for (int i = 1; i < numEnvs; i ++) {
    	auto resetResult = reset(*comms[i], 1);
    	obsvs[i] = std::get<0>(resetResult);
    	spdlog::info("Get reset result {}", i);
    }

    const int hiddenSize = 64;
    LunarRnnNet3 net(inputSize, outputSize, hiddenSize);
    net.to(device);
    spdlog::info("Net created ");
	torch::optim::Adagrad optimizer(net.parameters(), torch::optim::AdagradOptions(adaLr));
//	torch::optim::RMSprop optimizer(net.parameters(), torch::optim::RMSpropOptions(rmsLr).eps(1e-8).alpha(0.99));
//    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(1e-4));


    Tensor returns = torch::zeros({numEnvs});
    RunningMeanStd returnsRms(1);
//    vector<RnnMeanType> rms(numEnvs);


    bool render = false;

    for (int epoch = 0; epoch < epochNum; epoch ++) {
    	vector<Tensor> hStates(numEnvs,  torch::zeros({1, 1, hiddenSize}));
//    	vector<Tensor> valueStates(numEnvs, torch::zeros({1, 1, hiddenSize}));

        std::vector<float> runningReward(numEnvs, 0);
        Tensor rsmReturnTensor = torch::zeros({numEnvs, 1});
        std::vector<float> rsmRewards(numEnvs, 0);

        vector<bool> dones(numEnvs, false);

    	bool allDone = false;
    	int step = 0;
    	while (!allDone) {
    		allDone = true;
    		for (int index = 0; index < numEnvs; index ++) {
    			if (!store.isDone(storeIndex[index])) {
    				allDone = false;
    				auto netOutput = net.getAction(torch::tensor(obsvs[index]).view({1, 1, 8}), hStates[index]);
    				int action = netOutput[0].item<long>();
    				hStates[index] = netOutput[1];
    				auto stepResult = getStepResult(*comms[index], action, render, obsvSpace.shape);
    				float reward = std::get<1>(stepResult);
    				dones[index] = std::get<2>(stepResult);

    				runningReward[index] += reward;
    				rsmRewards[index] = reward;

    				store.addStep(storeIndex[index], obsvs[index], action, reward);

    				obsvs[index] = std::get<0>(stepResult);
    			} else {
    				rsmRewards[index] = 0.0f;
    			}
    		}

    		Tensor rewardTensor = torch::tensor(rsmRewards).view({numEnvs, 1});
    		rsmReturnTensor = rsmReturnTensor * discountFactor + rewardTensor;
    		returnsRms->update(rsmReturnTensor);
    		rewardTensor = torch::clamp(rewardTensor / torch::sqrt(returnsRms->get_variance() + 1e-8), -10, 10);
    		float * rewardData = rewardTensor.data_ptr<float>();
    		for (int index = 0; index < numEnvs; index ++) {
    			if (! store.isDone(storeIndex[index])) {
    				int recordSize = store.datas[storeIndex[index]].reward.size() - 1;
    				store.datas[storeIndex[index]].reward[recordSize] = rewardData[index];
    			}
    		}

    		for (int index = 0; index < numEnvs; index ++) {
    			if (dones[index] && (!store.isDone(storeIndex[index]))) {
       				store.done(storeIndex[index]);
        			float* returnData = rsmReturnTensor.data_ptr<float>();
        			returnData[index] = 0.0f;
        			rsmRewards[index] = 0.0f;
        		}
    		}
    	}


    	if (store.getDataSize() >= miniBatch)
    	{
//    		spdlog::info("------------------------> To update network");
    		float aveReward = std::accumulate(runningReward.begin(), runningReward.end(), 0.0f);
    		aveReward /= miniBatch;
    		spdlog::info("Rewards({}): {}", epoch, aveReward);
    		plotReward(aveReward);
    		updateNet(net, optimizer, store);
    	}

    	for (int index = 0; index < numEnvs; index ++) {
    		storeIndex[index] = store.getNextSlot();
    	}

    }

}


/**************************************************** Main *********************************************/

#define Main 1
int main(int argc, char** argv) {
//	testSample();
	testLunar();
}
