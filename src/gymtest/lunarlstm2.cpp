/*
 * lunarlstm2.cpp
 *
 *  Created on: Jul 11, 2020
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




struct LunarGruNet: torch::nn::Module {
private:
	const unsigned int numInputs;
	const unsigned int numActOutput;
	torch::nn::Linear ah;
	torch::nn::Linear vh;
	torch::nn::GRU ah1;
	torch::nn::GRU vh1;
	torch::nn::Linear ao;
	torch::nn::Linear vo;

public:
	LunarGruNet(unsigned iNumInputs, unsigned int iNumActOutput, unsigned int hiddenSize)
		: numInputs(iNumInputs),
		  numActOutput(iNumActOutput),
		  ah(torch::nn::Linear(iNumInputs, hiddenSize)),
		  vh(torch::nn::Linear(iNumInputs, hiddenSize)),
		  ah1(torch::nn::GRU(torch::nn::GRUOptions(hiddenSize, hiddenSize).num_layers(1).batch_first(true))),
		  vh1(torch::nn::GRUOptions(hiddenSize, hiddenSize).num_layers(1).batch_first(true)),
		  ao(torch::nn::Linear(hiddenSize, numActOutput)),
		  vo(torch::nn::Linear(hiddenSize, 1))
	{
		register_module("ah", ah);
		register_module("vh", vh);
		register_module("ah1", ah1);
		register_module("vh1", vh1);
		register_module("ao", ao);
		register_module("vo", vo);

		init_weights(ah->named_parameters(), sqrt(2.0), 0);
		init_weights(vh->named_parameters(), sqrt(2.0), 0);
		init_weights(ah1->named_parameters(), 1, 0);
		init_weights(vh1->named_parameters(), 1, 0);
		init_weights(ao->named_parameters(), sqrt(2.0), 0);
		init_weights(vo->named_parameters(), sqrt(2.0), 0);
	}

	//inputs sorted
	//batch, step, others
	vector<Tensor> forward(vector<Tensor> inputs) {
		vector<int> seqLens(inputs.size(), 0);
		const int seqLen = inputs[0].size(0);
		for (int i = 0; i < inputs.size(); i ++) {
			seqLens[i] = inputs[i].size(0);
			inputs[i] = inputs[i].view({inputs[i].size(0), -1}); //TODO: -1?
			inputs[i] = torch::constant_pad_nd(inputs[i], {0, 0, 0, (seqLen - seqLens[i])});
			cout << "inputs[i]: " << inputs[i].sizes() << endl;
		}

		Tensor packedInput = torch::stack(inputs, 0);

		Tensor actionOutput = ah->forward(packedInput);
		actionOutput = torch::leaky_relu(actionOutput);
		auto padActionInput = torch::nn::utils::rnn::pack_padded_sequence(actionOutput, torch::tensor(seqLens), true);
		auto actionGruOutputData = ah1->forward_with_packed_input(padActionInput);
		auto actionGruOutput = std::get<0>(actionGruOutputData);
		Tensor aoInput = actionGruOutput.data();
		Tensor aoOutput = ao->forward(aoInput);

		Tensor valueOutput = vh->forward(packedInput);
		valueOutput = torch::leaky_relu(valueOutput);
		auto padValueInput = torch::nn::utils::rnn::pack_padded_sequence(valueOutput, torch::tensor(seqLens), true);
		auto valueGruOutputData = vh1->forward_with_packed_input(padValueInput);
		auto valueGruOutput = std::get<0>(valueGruOutputData);
		Tensor voInput = valueGruOutput.data();
		Tensor voOutput = vo->forward(voInput);

		cout << "actionOutput: " << aoOutput.sizes() << endl;
		cout << "valueOutput: " << voOutput.sizes() << endl;

		//Seemed 2d tensors
		return {voOutput, aoOutput};
	}

	vector<Tensor> forward(Tensor input,  Tensor valueState, Tensor actionState) {
//		cout << "input: " << input.sizes() << endl;
//		cout << "actionState: " << actionState.sizes() << endl;
		Tensor actionOutput = ah->forward(input);
//		cout << "actionOutput: " << actionOutput.sizes() << endl;
		actionOutput = torch::leaky_relu(actionOutput);
		auto gruOutput = ah1->forward(actionOutput, actionState); //TODO: actionState
		actionOutput = std::get<0>(gruOutput);
		actionState = std::get<1>(gruOutput);
//		cout << "actionState: " << actionState.sizes() << endl;
		actionOutput = ao->forward(actionOutput);

		Tensor valueOutput = vh->forward(input);
		valueOutput = torch::leaky_relu(valueOutput);
		auto valueGruOutput = vh1->forward(valueOutput, valueState);
		valueOutput = std::get<0>(valueGruOutput);
		valueState = std::get<1>(valueGruOutput);
		valueOutput = vo->forward(valueOutput);

		return {valueOutput, actionOutput, valueState, actionState};
	}

/*	//TODO: No one calls?
	Tensor getActions(vector<Tensor> inputs) {
		this->eval();
		vector<Tensor> output = forward(inputs);
		Tensor actOutput = output[1];
		Tensor actProb = torch::softmax(actOutput, -1);
		actProb = actProb.clamp(1.21e-7, 1.0f - 1.21e-7);

		Tensor actions = actProb.multinomial(1, true);
		return actions;
	}*/

	float getNextValue(Tensor input, Tensor valueState, Tensor actionState) {
		this->eval();
		vector<Tensor> output = forward(input, valueState, actionState); //TODO: Should refresh state in each tuncation
		Tensor valueTensor = output[0];
		return valueTensor.item<float>();
	}

	vector<Tensor> getAction(Tensor input, Tensor valueState, Tensor actionState) {
		this->eval();
		vector<Tensor> output = forward(input, valueState, actionState);
		Tensor actionOutput = output[1].squeeze(1);
		Tensor actProb = torch::softmax(actionOutput, -1);
//		cout << "actProb: " << actProb.sizes() << endl;
		actProb = actProb.clamp(1.21e-7, 1.0f - 1.21e-7);

		Tensor action = actProb.multinomial(1, true);
		return {action, output[2], output[3] };
//		return {action.item<long>(), output[0].item<float>() };
	}

	Tensor getLoss(vector<Tensor> inputs, vector<Tensor> actions, vector<Tensor> actReturn) {
		//inputs[x]: seq, x
		//vector<> inputs: batch, seq, x
		auto sortFunc = [] (const Tensor& t0, const Tensor& t1) -> bool {
			return t0.size(0) > t1.size(0);
		};

		std::stable_sort(inputs.begin(), inputs.end(), sortFunc);
		std::stable_sort(actions.begin(), actions.end(), sortFunc);
		std::stable_sort(actReturn.begin(), actReturn.end(), sortFunc);

		this->train();

		vector<Tensor> output = forward(inputs);
		Tensor valueOutput = output[0];
		Tensor actionOutput = output[1];

		Tensor returnTensor = torch::cat(actReturn, 0);
		Tensor actionTensor = torch::cat(actions, 0);

		Tensor adv = returnTensor - valueOutput;
		Tensor valueLoss = 0.5 * adv.pow(2).mean();

//		cout << "inputs " << endl << inputs << endl;
//		cout << "output[1]: " << endl << output[1] << endl;
		Tensor actionLogProbs = torch::log_softmax(actionOutput, -1);
		Tensor actionProbs = torch::softmax(actionOutput, -1);
		actionProbs = actionProbs.clamp(1.21e-7, 1.0f - 1.21e-7);
		Tensor entropy = -(actionLogProbs * actionProbs).sum(-1).mean();
//		cout << "actionLogProbs: " << endl << actionLogProbs.sizes() << endl;
//		cout << "entropy: " << endl << (actionLogProbs * actionProbs).sum(-1).sizes() << endl;

		Tensor actPi = actionLogProbs.gather(-1, actionTensor);
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
//		cout << "entropy: " << entropy.item<float>() << endl;
		cout << "-----------------------------------------> " << endl;

		Tensor loss = valueLoss + actionLoss;
//		- entropy * 1e-4;
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
struct LstmStoreDataType {
	vector<vector<float> > obsv;
	std::vector<long> action;
	std::vector<float> reward;
	bool done;
	float nextValue;

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


		Tensor actionTensor = torch::zeros({stepSize, 1}, at::kLong);
		long* actionData = actionTensor.data_ptr<long>(); //TODO: Check action copied as long or int
		std::copy(action.begin(), action.end(), actionData);
//
//		Tensor rewardTensor = torch::zeros({stepSize, 1});
//		float* rewardData = rewardTensor.data_ptr<float>();
//		std::copy(reward.begin(), reward.end(), rewardData);

		Tensor returnTensor = torch::zeros({stepSize, 1});
		float* returnData = returnTensor.data_ptr<float>();
		float lastReturn = nextValue;
		for (int i = 0; i < stepSize; i ++) {
			returnData[i] = lastReturn * discountFactor + reward[i];
			lastReturn = returnData[i];
		}

		return {obsvTensor, actionTensor, returnTensor};
	}

	void setDone(float value) {
		done = true;
		nextValue = value;
	}

	bool isDone() {
		return done;
	}

	void reset() {
		done = false;
		obsv.clear();
		action.clear();
		reward.clear();
	}
};

struct LunarLstmStorageType {
	const int Cap = 64;
	vector<LstmStoreDataType> datas;
	int lastIndex; //The last index had been written
	int index; //The index to be written

	LunarLstmStorageType(): datas(vector<LstmStoreDataType>(Cap)),
			lastIndex(0),
			index(1)
	{
	}

	void addStep(int index, vector<float> ob, int action, float reward) {
		datas[index].addStep(std::move(ob), action, reward);
	}

	void done(int index, float nextValue) {
		datas[index].setDone(nextValue);
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

	std::pair<vector<vector<Tensor>>, vector<float>> getData() {
		vector<vector<Tensor>> dataTensor(3);
		vector<float> nextValues;
		int curIndex = (lastIndex + 1) % Cap;
		while (curIndex != index) {
			if (datas[curIndex].reward.size() > 0) {
				vector<Tensor> data = datas[curIndex].getData();
				dataTensor[0].push_back(data[0]);
				dataTensor[1].push_back(data[1]);
				dataTensor[2].push_back(data[2]);
				nextValues.push_back(datas[curIndex].nextValue);
			}
			datas[curIndex].reset();

			curIndex = (curIndex + 1) % Cap;
		}

		lastIndex = (index - 1 + Cap) % Cap;

		return {dataTensor, nextValues};
	}

	//TODO: index
	int getDataSize() {
		return (index + Cap - lastIndex - 1) % Cap;
	}
};

/******************************************* Game Params *******************************************************/
#define GameParams 1
const int numEnvs = 4;
const string envName = "LunarLander-v2";
const float rmsLr = 1e-3;
const float adaLr = 1e-3;

const int batchSize = 40;
const float clipParam = 0.2;
const float entropyCoef = 1e-3;
const float rewardClip = 100;
const float valueLossCoef = 0.5;
const float tau = 0.9;

const int maxFrames = 4 * 10e5;
const int epochNum = 4096;
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

	//    spdlog::info("Get obsv vector: {}", obsvVec);
	//    spdlog::info("Get obsvShape: {}", obsShape);
	//    std::cout << "orig obsv " << std::endl << obsvTensor << std::endl;

	auto rawRewardVec = flattenVector(stepResult->real_reward);
//	spdlog::info("raw rewards: {}", rawRewardVec);
	float reward = rawRewardVec[0];
//	spdlog::info("return rewards: {}", reward);
	//    spdlog::info("rawRewardVec: {}", rawRewardVec);
	//    spdlog::info("rawRewards");


    auto rawDone = stepResult->done;
    bool done = rawDone[0][0];

    return {std::move(obsvVec), reward, done};
}


//For each env
static Tensor getReturns(Tensor rewards) {
	Tensor returns = torch::zeros(rewards.sizes());
	//TODO: How if internal memory not contiguous?
//	spdlog::info("getReturns");
//	spdlog::info("nextValues: {}", nextValues);
//	cout << "rewards " << endl << rewards << endl;

	auto returnData = returns.data_ptr<float>();
	auto rewardData = rewards.data_ptr<float>();
	const int step = rewards.size(0);

	returnData[step - 1] = rewardData[step - 1];
	for (int i = step - 2; i >= 0; i --) {
		returnData[i] = returnData[i + 1] * discountFactor + rewardData[i];
	}
	cout << "rewards: " << rewards.sizes() << endl;
//	cout << "returns: " << endl << returns << endl;

	return returns;
}

static Tensor getReturns(Tensor rewards, float nextValues) {
	Tensor returns = torch::zeros(rewards.sizes());
	//TODO: How if internal memory not contiguous?
//	spdlog::info("getReturns");
//	spdlog::info("nextValues: {}", nextValues);
//	cout << "rewards " << endl << rewards << endl;

	auto returnData = returns.data_ptr<float>();
	auto rewardData = rewards.data_ptr<float>();
	const int step = rewards.size(0);

	returnData[step - 1] = rewardData[step - 1] + nextValues * discountFactor;
	for (int i = step - 2; i >= 0; i --) {
		returnData[i] = returnData[i + 1] * discountFactor + rewardData[i];
	}
//	cout << "rewards: " << rewards.sizes() << endl;
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
static void updateNet(LunarGruNet& net, OptimzerType& optimizer, LunarLstmStorageType& storage) {
	//obsv: step, batch, others
	auto storageData = storage.getData();
	vector<vector<Tensor>> datas = std::get<0>(storageData);
	vector<Tensor> obsvs = datas[0];
	vector<Tensor> actions = datas[1];
	vector<Tensor> returns = datas[2];

//	vector<Tensor> returns(rewards.size());
//	for (int i = 0; i < returns.size(); i ++) {
//		returns[i] = getReturns(rewards[i]);
//	}

//	Tensor getLoss(Tensor inputs, Tensor actions, Tensor rewards, Tensor dones, vector<float> nextValues) {
//	Tensor loss = net.getLoss(obsv, actions, rewards, dones, nextValues);
	Tensor loss = net.getLoss(obsvs, actions, returns);

	optimizer.zero_grad();
	loss.backward();
	optimizer.step();
}

template <typename OptimzerType>
static void updateNet(LunarGruNet& net, OptimzerType& optimizer, LunarLstmStorageType& storage, bool calNextValue) {
	//obsv: step, batch, others
	auto storageData = storage.getData();
	vector<vector<Tensor>> datas = std::get<0>(storageData);
	vector<float> nextValues = std::get<1>(storageData);
	vector<Tensor> obsvs = datas[0];
	vector<Tensor> actions = datas[1];
	vector<Tensor> rewards = datas[2];

	vector<Tensor> returns(rewards.size());
	for (int i = 0; i < returns.size(); i ++) {
		returns[i] = getReturns(rewards[i], nextValues[i]);
	}

//	Tensor getLoss(Tensor inputs, Tensor actions, Tensor rewards, Tensor dones, vector<float> nextValues) {
//	Tensor loss = net.getLoss(obsv, actions, rewards, dones, nextValues);
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
    	std::string serverAddr = "tcp://127.0.0.1:1020" + std::to_string(i + 1);
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
	const int inputSize = obsvs[0].size();
	const int outputSize = actionSpace.shape[actionSpace.shape.size() - 1];
	spdlog::info("Get reset result {}", 0);
    for (int i = 1; i < numEnvs; i ++) {
    	auto resetResult = reset(*comms[i], 1);
    	obsvs[i] = std::get<0>(resetResult);
    	spdlog::info("Get reset result {}", i);
    }

    const int hiddenSize = 64;
    LunarGruNet net(inputSize, outputSize, hiddenSize);
    net.to(device);
    spdlog::info("Net created ");
	torch::optim::Adagrad optimizer(net.parameters(), torch::optim::AdagradOptions(adaLr));
//	torch::optim::RMSprop optimizer(net.parameters(), torch::optim::RMSpropOptions(rmsLr).eps(1e-8).alpha(0.99));
//    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(1e-4));


//    Tensor returns = torch::zeros({numEnvs});
//    Tensor doneTensor = torch::zeros({numEnvs, 1});
//    RunningMeanStd returnsRms(1);


    const int rewardStoreCap = 40;
    std::vector<float> rewardStore(rewardStoreCap, 0);
    int episode = 0;
    bool render = false;
    float runningReward = 0;

    for (int epoch = 0; epoch < epochNum; epoch ++) {
    	vector<Tensor> actionStates(numEnvs,  torch::zeros({1, 1, hiddenSize}));
    	vector<Tensor> valueStates(numEnvs, torch::zeros({1, 1, hiddenSize}));

    	for (int step = 0; step < updateInterval; step ++) {
    		for (int index = 0; index < numEnvs; index ++) {
    			auto netOutput = net.getAction(torch::tensor(obsvs[index]).view({1, 1, 8}), valueStates[index], actionStates[index]);
    			int action = netOutput[0].item<long>();
    			valueStates[index] = netOutput[1];
    			actionStates[index] = netOutput[2];

    			auto stepResult = getStepResult(*comms[index], action, render, obsvSpace.shape);
    			float reward = std::get<1>(stepResult);
    			reward = (reward > 10? 10 :reward) < -10? -10: reward;
    			runningReward += reward;
    			store.addStep(storeIndex[index], obsvs[index], action, reward); //TODO: obsvs[index] deep cloned?

    			obsvs[index] = std::get<0>(stepResult);
    			if (std::get<2>(stepResult)) {
    				store.done(storeIndex[index], 0);
    				storeIndex[index] = store.getNextSlot();
    				actionStates[index] = torch::zeros({1, 1, hiddenSize});
    				valueStates[index] = torch::zeros(actionStates[index].sizes());
    			}
    		}
    	}

    	for (int index = 0; index < numEnvs; index ++) {
    		float nextValue = net.getNextValue(torch::tensor(obsvs[index]).view({1, 1, 8}), valueStates[index], actionStates[index]);
    		store.done(storeIndex[index], nextValue);
    	}
    	float aveReward = runningReward / store.getDataSize();
    	spdlog::info("----------------------> {}", aveReward);
    	plotReward(aveReward);
    	runningReward = 0;

    	updateNet(net, optimizer, store, true);

    	for (int index = 0; index < numEnvs; index ++) {
    		valueStates[index] = torch::zeros({1, 1, hiddenSize});
    	    actionStates[index] = torch::zeros(valueStates[index].sizes());
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
