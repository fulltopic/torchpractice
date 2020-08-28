/*
 * lunarlstm4.cpp
 *
 *  Created on: Jul 14, 2020
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
	torch::nn::GRU ag;
	torch::nn::GRU vg;
	torch::nn::Linear ao;
	torch::nn::Linear vo;

public:
	LunarGruNet(unsigned iNumInputs, unsigned int iNumActOutput, unsigned int hiddenSize)
		: numInputs(iNumInputs),
		  numActOutput(iNumActOutput),
		  ah(torch::nn::Linear(hiddenSize, hiddenSize)),
		  vh(torch::nn::Linear(hiddenSize, hiddenSize)),
		  ag(torch::nn::GRU(torch::nn::GRUOptions(iNumInputs, hiddenSize).num_layers(1).batch_first(true))),
		  vg(torch::nn::GRUOptions(iNumInputs, hiddenSize).num_layers(1).batch_first(true)),
		  ao(torch::nn::Linear(hiddenSize, numActOutput)),
		  vo(torch::nn::Linear(hiddenSize, 1))
	{
		register_module("ag", ag);
		register_module("vg", vg);
		register_module("ah", ah);
		register_module("vh", vh);
		register_module("ao", ao);
		register_module("vo", vo);

		init_weights(ah->named_parameters(), sqrt(2.0), 0);
		init_weights(vh->named_parameters(), sqrt(2.0), 0);
		init_weights(ag->named_parameters(), 1, 0);
		init_weights(vg->named_parameters(), 1, 0);
		init_weights(ao->named_parameters(), sqrt(2.0), 0);
		init_weights(vo->named_parameters(), sqrt(2.0), 0);
	}

	//inputs sorted
	//batch, step, others
	//Does the output match action in loss function?
	vector<Tensor> forward(vector<Tensor> inputs) {
		Tensor inputTensor = torch::stack(inputs, 0);

		auto actionGruOutput = ag->forward(inputTensor);
		Tensor actionOutput = std::get<0>(actionGruOutput);
		actionOutput = torch::leaky_relu(actionOutput);
		actionOutput = ah->forward(actionOutput);
		actionOutput = torch::leaky_relu(actionOutput);
		actionOutput = ao->forward(actionOutput);

		auto valueGruOutput = vg->forward(inputTensor);
		Tensor valueOutput = std::get<0>(valueGruOutput);
		valueOutput = torch::leaky_relu(valueOutput);
		valueOutput = vh->forward(valueOutput);
		valueOutput = torch::leaky_relu(valueOutput);
		valueOutput = vo->forward(valueOutput);


		cout << "actionOutput: " << actionOutput.sizes() << endl;
		cout << "valueOutput: " << valueOutput.sizes() << endl;

		//Seemed 3d tensors
		return {valueOutput, actionOutput};
	}

	//TODO: input {1, 1, -1}
	vector<Tensor> forward(Tensor input,  Tensor valueState, Tensor actionState) {
		auto actionGruOutput = ag->forward(input, actionState);
		Tensor actionOutput = std::get<0>(actionGruOutput);
		Tensor newActionState = std::get<1>(actionGruOutput);
		actionOutput = ah->forward(actionOutput);
		actionOutput = torch::relu(actionOutput);
		actionOutput = ao->forward(actionOutput);

		auto valueGruOutput = vg->forward(input, valueState);
		Tensor valueOutput = std::get<0>(valueGruOutput);
		Tensor newValueState = std::get<1>(valueGruOutput);
		valueOutput = vh->forward(valueOutput);
		valueOutput = torch::relu(valueOutput);
		valueOutput = vo->forward(valueOutput);

		return {valueOutput, actionOutput, newValueState, newActionState};
	}


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

		Tensor action = actProb.multinomial(1, false);
		return {action, output[2], output[3] };
	}

	Tensor getLoss(vector<Tensor> inputs, vector<Tensor> actions, vector<Tensor> actReturn) {
		this->train();

		vector<Tensor> output = forward(inputs);
		Tensor valueOutput = output[0];
		Tensor actionOutput = output[1];

		Tensor returnTensor = torch::stack(actReturn, 0);
		Tensor actionTensor = torch::stack(actions, 0);

		Tensor adv = returnTensor - valueOutput;
		Tensor valueLoss = 0.5 * adv.pow(2).mean();

		Tensor actionLogProbs = torch::log_softmax(actionOutput, -1);
		Tensor actionProbs = torch::softmax(actionOutput, -1);
		actionProbs = actionProbs.clamp(1.21e-7, 1.0f - 1.21e-7);
		Tensor entropy = -(actionLogProbs * actionProbs).sum(-1).mean();
//		cout << "actionLogProbs: " << endl << actionLogProbs << endl;
//		cout << "entropy: " << endl << (actionLogProbs * actionProbs).sum(-1).sizes() << endl;

		Tensor actPi = actionLogProbs.gather(-1, actionTensor);
//		cout << "actPi: " << endl << actPi << endl;
//		cout << "adv: " << endl << adv.sizes() << endl;
//		cout << "actions: " << endl << actionTensor << endl;
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

		Tensor loss = valueLoss + actionLoss - entropy * (1e-4);
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
	std::vector<bool> dones;
	float nextValue;

	void addStep(vector<float> ob, long act, float score, bool isDone) {
		obsv.push_back(ob);
		action.push_back(act);
		reward.push_back(score);
		dones.push_back(isDone);
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

		Tensor returnTensor = torch::zeros({stepSize, 1});
		float* returnData = returnTensor.data_ptr<float>();
		float lastReturn = nextValue;
		for (int i = stepSize - 1; i >= 0; i --) {
			if (dones[i]) {
				lastReturn = 0.0f;
			}
			returnData[i] = lastReturn * discountFactor + reward[i];
			lastReturn = returnData[i];
		}
//		spdlog::info("rewards: {}", reward);
//		cout << "returns: " << endl << returnTensor << endl;

		return {obsvTensor, actionTensor, returnTensor};
	}

	void setDone(float value) {
		nextValue = value;
	}

	int size() {
		return dones.size();
	}

	void reset() {
		obsv.clear();
		action.clear();
		reward.clear();
		dones.clear();
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

	void addStep(int index, vector<float> ob, int action, float reward, bool isDone) {
		datas[index].addStep(std::move(ob), action, reward, isDone);
	}

	void done(int index, float nextValue) {
		datas[index].setDone(nextValue);
	}

	int getSize(int index) {
		return datas[index].size();
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
const int epochNum = 2048 * 4;
const int miniBatch = 8;
const int updateInterval = 64;
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

	auto rawRewardVec = flattenVector(stepResult->real_reward);
	float reward = rawRewardVec[0];


    auto rawDone = stepResult->done;
    bool done = rawDone[0][0];

    return {std::move(obsvVec), reward, done};
}


/********************************************* Training *************************************************************/
#define TRAIN 1

template <typename OptimzerType>
static void updateNet(LunarGruNet& net, OptimzerType& optimizer, LunarLstmStorageType& storage) {
	//obsv: step, batch, others
	auto datas = storage.getData();
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
    	std::string serverAddr = "tcp://127.0.0.1:1020" + std::to_string(i + 1);
    	spdlog::info("Connecting to gym server {}", serverAddr);
    	comms.push_back(new Communicator(serverAddr));
    }

    vector<vector<float>> obsvs(numEnvs);

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


    const int rewardStoreCap = 40;
    std::vector<float> rewardStore(rewardStoreCap, 0);
    bool render = false;
    float runningReward = 0;

    //TODO: When to reinitialize states, after done? after update? not at all?
	vector<Tensor> actionStates(numEnvs,  torch::zeros({1, 1, hiddenSize}));
	vector<Tensor> valueStates(numEnvs, torch::zeros({1, 1, hiddenSize}));
    RunningMeanStd returnsRms(1);
    vector<float> rmsReturns(numEnvs, 0);
	vector<float> rmsRewards(numEnvs, 0);

    for (int epoch = 0; epoch < epochNum; epoch ++) {
    	for (int step = 0; step < updateInterval; step ++) {
    		for (int index = 0; index < numEnvs; index ++) {
    			auto netOutput = net.getAction(torch::tensor(obsvs[index]).view({1, 1, 8}), valueStates[index], actionStates[index]);
    			int action = netOutput[0].item<long>();
    			valueStates[index] = netOutput[1];
    			actionStates[index] = netOutput[2];

    			auto stepResult = getStepResult(*comms[index], action, render, obsvSpace.shape);
    			float reward = std::get<1>(stepResult);
    			bool isDone = std::get<2>(stepResult);
    			runningReward += reward;
    			store.addStep(storeIndex[index], obsvs[index], action, reward, isDone); //TODO: obsvs[index] deep cloned?

    			rmsRewards[index] = reward;
    			obsvs[index] = std::get<0>(stepResult);

    			if (isDone) {
    				actionStates[index] = torch::zeros({1, 1, hiddenSize});
    				valueStates[index] = torch::zeros(actionStates[index].sizes());
    				rmsReturns[index] = 0.0f;
    			}
    		}

//    		cout << "rewardTensor: " << torch::tensor(rmsRewards.data()).sizes() << endl;
    		Tensor rewardTensor = torch::tensor(rmsRewards).view({numEnvs, 1});
    		Tensor returnTensor = torch::tensor(rmsReturns).view({numEnvs, 1});
    		returnTensor = returnTensor * discountFactor + rewardTensor;
    		returnsRms->update(returnTensor);
    		rewardTensor = torch::clamp(rewardTensor / torch::sqrt(returnsRms->get_variance() + 1e-8), -10, 10);
    		float* rmsReturnData = returnTensor.data_ptr<float>();
    		float* rmsRewardData = rewardTensor.data_ptr<float>();
    		for (int index = 0; index < numEnvs; index ++) {
    			rmsReturns[index] = rmsReturnData[index];
    			rmsRewards[index] = rmsRewardData[index];
    		}
//    		cout << "rewardTensor: " << rewardTensor << endl;
//    		spdlog::info("reward vector: {}", rmsRewards);
    		//TODO: check vector updated


    		int currStep = store.getSize(storeIndex[0]) - 1;
    		for (int index = 0; index < numEnvs; index ++) {
    			store.datas[storeIndex[index]].reward[currStep] = rmsRewards[index];
    		}
    	}


		int stepSize = store.getSize(storeIndex[0]);
		for (int index = 0; index < numEnvs; index ++) {
    		bool isDone = store.datas[storeIndex[index]].dones[stepSize];
    		float nextValue = 0.0f;
    		if (!isDone) {
    			nextValue = net.getNextValue(torch::tensor(obsvs[index]).view({1, 1, 8}), valueStates[index], actionStates[index]);
    		}
    		store.done(storeIndex[index], nextValue);
    	}

    	float aveReward = runningReward / store.getDataSize();
    	plotReward(aveReward);
    	spdlog::info("----------------------> {}", aveReward);
//    	if (aveReward > -120) {
//    		render = true;
//    	} else {
//    		render = false;
//    	}
    	runningReward = 0;

    	updateNet(net, optimizer, store);

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
