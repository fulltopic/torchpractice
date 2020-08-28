/*
 * testrnnlen.cpp
 *
 *  Created on: May 24, 2020
 *      Author: zf
 */




#include <torch/torch.h>
#include <torch/script.h>

#include "lmdbtools/LmdbReader.h"
#include "lmdbtools/LmdbDataDefs.h"
#include "lmdbtools/Lmdb2RowDataDefs.h"
#include "lmdbtools/LmdbReaderWrapper.h"
//#include "NetDef.h"
#include <matplotlibcpp.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <stdio.h>
#include <bits/stdc++.h>
#include <sys/types.h>
#include <experimental/filesystem>
#include <thread>
#include <cmath>
#include <time.h>
#include <ctime>
#include <chrono>
#include <functional>


#include "nets/lstmnet.h"
#include "nets/grunet.h"
#include "nets/fcgrunet.h"
#include "nets/cnngrunet.h"
#include "nets/lstm2net.h"
#include "nets/rnnmasknet.h"

#include "pytools/syncplotserver.h"

#include <boost/lockfree/queue.hpp>


using Tensor = torch::Tensor;
using TensorList = torch::TensorList;
using string = std::string;

using std::endl;
using std::cout;

using DbDataType = std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>>;

const std::string wsPath = "/home/zf/";
std::ofstream dataFile(wsPath + "/workspaces/workspace_cpp/torchpractice/build/errorstats.txt");
const int batchSize = 128;

std::function<void(int)> shutdownHandler;
void signalHandler(int signal) {
	std::cout << "Received signal " << signal << std::endl;
	shutdownHandler(signal);
	exit(0);
}

static bool compTensorBySeqLen (const Tensor& t0, const Tensor& t1) {
	return t0.size(0) > t1.size(0);
}

static void processActionInput(std::vector<Tensor>& inputs, std::vector<Tensor>& labels) {
//	cout << inputs[0].sizes() << std::endl;
//	cout << labels[0].sizes() << std::endl;
	for (int i = 0; i < inputs.size(); i ++) {
		auto inputPtr = inputs[i].accessor<float, 3>();
		auto labelPtr = labels[i].data_ptr<long>();
		for (int j = 0; j < labels[i].size(0); j ++) {
			if (labelPtr[j] >= 34) {
//				cout << "Label is -------------------------> " << labelPtr[j] << endl;
				for (int k = 34; k < 42; k ++) {
					inputPtr[j][1][k] = 1;
				}
//				cout << "Update input: " << inputs[i][j] << endl;
			}
		}
	}
	std::stable_sort(inputs.begin(), inputs.end(), compTensorBySeqLen);
	std::stable_sort(labels.begin(), labels.end(), compTensorBySeqLen);
}

static float evaluation(Tensor outputs, Tensor labels, bool isTest) {
//	cout << "Evaluation output sizes: " << outputs.sizes() << endl;
//	cout << "Evaluation label sizes: " << labels.sizes() << endl;
//	const int batchSize = outputs.size(0);
	Tensor values;
	Tensor indices;
	indices = torch::argmax(outputs, 1);
//	cout << "Evaluation indices sizes: " << indices.sizes() << endl;

	Tensor diff = torch::sub(labels, indices);
	int total = labels.size(0);
	auto matched = total - diff.nonzero().size(0);
//
//	std::cout << "Max " << indices.sizes() << std::endl;
//	std::cout << "Max " << indices[0] << std::endl;
//	std::cout << outputs[0] << std::endl;
	const long* labelData = labels.data_ptr<long>();
	const long* indexData = indices.data_ptr<long>();

//	int matched = 0;
	if (isTest) {
		std::vector<std::vector<int>> errCounts(42, std::vector<int>(42, 0));
		std::vector<int> labelErrs(42, 0);
		std::vector<int> labelNums(42, 0);
		for (int i = 0; i < total; i ++) {
			labelNums[labelData[i]] ++;
			if (labelData[i] != indexData[i]){
				errCounts[(int)labelData[i]][(int)indexData[i]] ++;
				labelErrs[(int)labelData[i]] ++;
			}
		}
		dataFile << "------------------------------------------------------------> Evaluation " << std::endl;
		for (int i = 0; i < errCounts.size(); i ++) {
			dataFile << "[" << labelErrs[i] << "/" << labelNums[i] << " = " << (float)labelErrs[i] / labelNums[i] << "]" << ": ";
			for (int j = 0; j < errCounts[0].size(); j ++) {
				dataFile << errCounts[i][j] << ", ";
			}
			dataFile << endl;
		}
	}

	float accu = (float)matched / total;
//	std::cout << "Accuracy: " << matched << "/" << total << " = " << (float)matched / total << std::endl;
//
//	for (int i = 0; i < seqLen; i ++) {
//		std::cout << indexData[i] << "-->" << labelData[i] << ", ";
//	}
//	std::cout << std::endl;

	return accu;
}


template<typename NetType>
static std::pair<float, float> testLstmOverfit(std::vector<Tensor>& inputs, std::vector<Tensor>& labels,
		const int seqLen, NetType& net)
{
	net.eval();

	Tensor labelTensor = torch::cat(labels, 0);
	Tensor output = net.forward(inputs, seqLen, false, false);

	auto loss = torch::nll_loss(output, labelTensor).item<float>();
	auto accu = evaluation(output, labelTensor, true);

	return std::make_pair(loss, accu);
}

static Tensor createLabelTensor(std::vector<Tensor>& labels, const int seqLen) {
	int count = 0;
	for (int i = 0; i < labels.size(); i ++) {
		if (labels[i].size(0) > seqLen) {
			labels[i] = labels[i].narrow(0, 0, seqLen);
//			std::cout << "narrowed: " << labels[i].size(0) << std::endl;
		} else {
//			std::cout << "remain: " << labels[i].size(0) << std::endl;
		}
		count += (int)labels[i].size(0);
	}
//	std::cout << "Total ------------------------------> " << count << std::endl;

	return torch::cat(labels, 0);
}

template<typename NetType>
static std::pair<float, float> validLstmOverfitWithPlot(std::vector<Tensor>& inputs, std::vector<Tensor>& labels,
		const int seqLen, NetType& net, SyncPlotServer& plotServer, std::vector<float> params)
{
	net.eval();

//	Tensor labelTensor = torch::cat(labels, 0);
	Tensor labelTensor = createLabelTensor(labels, seqLen);
	Tensor output = net.forward(inputs, seqLen, false, false);

	auto loss = torch::nll_loss(output, labelTensor).item<float>();
	auto accu = evaluation(output, labelTensor, true);

	plotServer.validUpdate(output, labelTensor, params);
	plotServer.refresh();

	return std::make_pair(loss, accu);
}

template<typename NetType, typename PlotServerType>
static void saveOutput(NetType& net, PlotServerType& plotServer, const int sampleNum, const float lr) {
	auto saveTime = std::chrono::system_clock::now().time_since_epoch();
	auto saveSecond = std::chrono::duration_cast<std::chrono::seconds>(saveTime).count();
	std::string fileName = net.GetName() + "_" + std::to_string(sampleNum) + "_" + std::to_string(lr) + "_" + std::to_string(saveSecond);
	plotServer.save(fileName + ".png");
	std::string modelPath = fileName + ".pt";
//	plotServer.save(net.GetName() + "_" + (std::to_string(saveSecond)));
//	std::string modelPath = net.GetName() + "_" + (std::to_string(saveSecond)) + ".pt";
	torch::serialize::OutputArchive output_archive;
	net.save(output_archive);
	output_archive.save_to(modelPath);
}


template<typename NetType>
static void test5RowsDbsMask(const float lr, const int sampleNum, const int seqLen) {
	std::string dbPath = wsPath + "/workspaces/res/dbs/lmdb5rowscenetest";

	LmdbSceneReader<LmdbDataDefs> reader(dbPath);

	std::string validDbPath = wsPath + "/workspaces/res/dbs/lmdb5rowscenetestvalid";
	LmdbSceneReader<LmdbDataDefs> validReader(validDbPath);

	NetType net(seqLen);
	torch::optim::Adagrad optimizer(net.parameters(), torch::optim::AdagradOptions(lr));

	trainLstmDbOverfit(reader, validReader, net, optimizer, sampleNum, seqLen);
}

static const int VecCap = 128;
static std::vector<DbDataType> datas(VecCap);
boost::lockfree::queue<int> cq(VecCap);
boost::lockfree::queue<int> pq(VecCap);
static volatile bool readerWork = true;
static volatile bool toReset = false;

static void dbReaderFunc (LmdbSceneReader<LmdbDataDefs>& reader, const int batchSize) {
	while (readerWork) {
//		if (toReset || (!reader.hasNext())) {
//			reader.reset();
//			toReset = false;
//		}
		if ((!toReset) && (!reader.hasNext())) {
			sleep(1);
		}
		if (toReset) {
			reader.reset();
			toReset = false;
		}
		DbDataType data = reader.next(batchSize);
		std::vector<Tensor> inputs;
		std::vector<Tensor> labels;
		std::tie(inputs, labels) = data;
		processActionInput(inputs, labels);
		data = std::make_pair(inputs, labels);

		int index = -1;
		while ((!pq.pop(index)) && readerWork) {
			sleep(1);
		}

		if (index >= 0) {
			datas[index] = data;
			cq.push(index);
		}
	}
}

static void cleanQ() {
	readerWork = false;
	int dummy = 0;
	while (cq.pop(dummy));
	while (pq.pop(dummy));
}

//typedef std::function<void(int)> SigHandleType;
typedef void (*SigHandleType) (int);

static SigHandleType regSigHandler(SigHandleType handler) {
	auto origHandler = std::signal(SIGABRT, handler);
	std::signal(SIGFPE, handler);
	std::signal(SIGILL, handler);
	std::signal(SIGINT, handler);
	std::signal(SIGSEGV, handler);
	std::signal(SIGTERM, handler);

	return origHandler;
}


struct ValidStats {
//	const int sampleNum;

//	double totalTrainLoss;
//	double totalTrainAccu;
	int iteNum;
	double totalValidTrainLoss;
	double totalValidTrainAccu;

	float minValidTrainLoss;
	float maxValidTrainAccu;
	float aveValidTrainLoss;
	float aveValidTrainAccu;

	ValidStats(int inSampleNum):
		iteNum(0),
//		sampleNum(inSampleNum),
//			totalTrainLoss(0),
//			totalTrainAccu(0),
			totalValidTrainLoss(0),
			totalValidTrainAccu(0),
			minValidTrainLoss(0),
			maxValidTrainAccu(0),
			aveValidTrainLoss(0),
			aveValidTrainAccu(0)
	{
	}
	~ValidStats() {}

	void reset() {
//		totalTrainLoss = 0;
//		totalTrainAccu = 0;
		iteNum = 0;
		totalValidTrainLoss = 0;
		totalValidTrainAccu = 0;
		minValidTrainLoss = FLT_MAX;
		maxValidTrainAccu = 0;
		aveValidTrainLoss = 0;
		aveValidTrainAccu = 0;
	}

	void updateTrain(float trainLoss, float trainAccu) {
//		cout << "Update train " << trainLoss << ", " << trainAccu << endl;
		totalValidTrainLoss += trainLoss;
		totalValidTrainAccu += trainAccu;
		if (trainLoss < minValidTrainLoss) {
			minValidTrainLoss = trainLoss;
//			cout << "Update minValidTrainLoss " << minValidTrainLoss << endl;
		}
		if (trainAccu > maxValidTrainAccu) {
			maxValidTrainAccu = trainAccu;
//			cout << "Update maxValidTrainAccu " << maxValidTrainAccu << endl;
		}
		iteNum ++;
//		cout << "iteNum = " << iteNum << endl;
//		cout << "totalValidTrainLoss = " << totalValidTrainLoss << endl;
//		cout << "totalValidTrainAccu = " << totalValidTrainAccu << endl;
	}

	std::vector<float> getValidStats() {
		if (iteNum > 0) {
			aveValidTrainLoss = totalValidTrainLoss / iteNum;
			aveValidTrainAccu = totalValidTrainAccu / iteNum;
		}
		return {minValidTrainLoss, aveValidTrainLoss, maxValidTrainAccu, aveValidTrainAccu};
	}
};

template<typename NetType, typename DbDefs>
static void trainLstmDbOverfit(
		LmdbSceneReader<DbDefs>& validReader,
		NetType& net, torch::optim::Optimizer& optimizer, const int sampleNum, const int seqLen, const float lr) {

	int64_t totalNum = 0;
	double totalLoss = 0;
	const int epoch = 256;
//	const int validStep = 128;
	const int earlyStop = epoch / 8;
	const float validRatio = 0.001;
	int stopStep = 0;
	float lastLoss = FLT_MAX;

	std::map<std::string, std::string> keys;

	SyncPlotServer plotServer(net.parameters().size(), net.parameters(), "Sample# " + std::to_string(sampleNum), true);
	ValidStats validStater(sampleNum);

	shutdownHandler = [&](int signal) {
		cleanQ();
		saveOutput(net, plotServer, sampleNum, lr);
	};
	auto origHandler = regSigHandler(signalHandler);


	std::vector<Tensor> validInputs;
	std::vector<Tensor> validLabels;

	int validSetSize = std::max(256, (int)(sampleNum * validRatio));
	validSetSize = batchSize * 2; //TODO: hard-coded
//	cout << "Get validation dataset with db vector: " << validInputs.size() << endl;

	int iteNum = 1;

	for (int count = 0; count <= epoch; count ++) {
		totalNum = 0;
		totalLoss = 0;

		validStater.reset();
		bool printed = false;
		int readerIndex = 0;
		toReset = true;

		net.train();
		while (totalNum < sampleNum) {
			std::vector<Tensor> inputs;
			std::vector<Tensor> labels;
			int index = -1;
			optimizer.zero_grad();

			while (!cq.pop(index)) {

				sleep(1);
			}

			auto data = datas[index];
			std::tie(inputs, labels) = data;
			totalNum += inputs.size();
//			for (int i = 0; i < inputs.size(); i ++) {
//				cout << "------------> " << inputs[i].sizes() << endl;
//			}


			Tensor output = net.forward(inputs, seqLen, true, false);
//
//
//			for (int i = 0; i < labels.size(); i ++) {
//				if (labels[i].size(0) > seqLen) {
//					labels[i] = labels[i].narrow(0, 0, seqLen);
//				}
//			}
			Tensor labelTensor = createLabelTensor(labels, seqLen);
//			for (int i = 0; i < labels.size(); i ++) {
//				cout << "labels: " << labels[i].size(0) << endl;
//			}
			auto loss = torch::nll_loss(output, labelTensor);
			float lossValue = loss.item<float>();
			totalLoss +=(lossValue * inputs.size());
			auto accu = evaluation(output, labelTensor, false);
//			totalTrainAccu += accu;
			std::cout << "====================================================> loss " << loss.item<float>()
				<< ", " << (totalLoss / totalNum)
				<< ", " << accu << std::endl;

			loss.backward();
			optimizer.step();

			plotServer.trainUpdate(output, labelTensor, net.parameters());
			iteNum = (iteNum + 1) % 65535;
			if (iteNum % 32 == 0) {
				plotServer.refresh();
			}

			pq.push(index);

			validStater.updateTrain(lossValue, accu);
		}

		//Validation
		{
			validReader.reset();
			std::tie(validInputs, validLabels) = validReader.next(validSetSize);
			processActionInput(validInputs, validLabels);

			auto params = validStater.getValidStats();
			auto validLoss = validLstmOverfitWithPlot(validInputs, validLabels, seqLen, net, plotServer, params);
			//		cout << "Valid input " << endl;
			//		cout << validInputs[0][0] << endl;
			std::cout << "---------------------------------> " << count << " Validation loss: " << std::get<0>(validLoss)
					<< ", " << std::get<1>(validLoss) << std::endl;


			if (std::get<0>(validLoss) < lastLoss) {
				stopStep = 0;
				lastLoss = std::get<0>(validLoss);
			} else {
				stopStep ++;
				if (stopStep > earlyStop) {
					std::cout << "Early stop " << std::endl;
					break;
				}
			}
		}
	}

	std::cout << "End of training " << std::endl;
	cleanQ();
	std::cout << "End of queues " << std::endl;
	saveOutput(net, plotServer, sampleNum, lr);
	regSigHandler(origHandler);
}

template<typename NetType>
static void testTrainByQ(float lr, int sampleNum, int seqLen) {
	std::string dbPath = wsPath + "/workspaces/res/dbs/lmdb5rowscenetest";
	LmdbSceneReader<LmdbDataDefs> reader(dbPath);
	std::string validDbPath = wsPath + "/workspaces/res/dbs/lmdb5rowscenetestvalid";
	LmdbSceneReader<LmdbDataDefs> validReader(validDbPath);

	NetType net(seqLen);
	torch::optim::Adagrad optimizer(net.parameters(), torch::optim::AdagradOptions(lr));
//	torch::optim::Adam optimizer(net.parameters(), torch::optim::AdagradOptions(lr));

	for (int i = 0; i < VecCap; i ++) {
		pq.push(i);
	}

	std::thread t(dbReaderFunc, std::ref(reader), batchSize);

	trainLstmDbOverfit(validReader, net, optimizer, sampleNum, seqLen, lr);

	readerWork = false;
	t.join();
}


static void testDataQ() {
	boost::lockfree::queue<int> pq(VecCap);
	boost::lockfree::queue<int> cq(VecCap);
	for (int i = 0; i < VecCap; i ++) {
		pq.push(i);
	}
	const int batchSize = 2;
	std::string dbPath = wsPath + "/workspaces/res/dbs/lmdb5rowscenetest";
	LmdbSceneReader<LmdbDataDefs> reader(dbPath);

	std::thread t(dbReaderFunc, std::ref(reader), batchSize);

	for (int i = 0; i < 20; i ++) {
		int index = -1;
		while (!cq.pop(index)) {
			sleep(1);
		}
		auto data = datas[index];
		auto inputs = std::get<0>(data);
		auto labels = std::get<1>(data);
		cout << "Inputs " << inputs.size() << endl;
		cout << inputs[0].sizes() << endl;
		auto dataPtr = inputs[0].accessor<float, 3>();
		for (int j = 0; j < inputs[0].size(0); j ++) { //seqLen
			if (dataPtr[j][1][34] == 1) {
				cout << "Maybe " << endl;
				for (int k = 34; k < 42; k ++) {
					if (dataPtr[j][1][k] != 1) {
						cout << "Failed " << endl;
					}
				}
				cout << inputs[0][j] << endl;
				cout << "-----------> " << endl;
			}
		}

		pq.push(index);

	}

	readerWork = false;
	t.join();
}

struct CompData {
	Tensor data;
	int index;
};

//static bool compCompData (const CompData& c0, const CompData& c1) {
//	return c0.data.size(0) > c1.data.size(0);
//}

static void testStableSort() {
	std::string dbPath = wsPath + "/workspaces/res/dbs/lmdb5rowscenetest";
	LmdbSceneReader<LmdbDataDefs> reader(dbPath);

	const int num = 32;
	auto datas = reader.next(num);
	std::vector<CompData> inputs;
	std::vector<CompData> labels;
	for (int i = 0; i < num; i ++) {
		inputs.push_back({std::get<0>(datas)[i], i});
		labels.push_back({std::get<1>(datas)[i], i});
	}

	auto compCompData = [] (const CompData& c0, const CompData& c1) -> bool {
		return c0.data.size(0) > c1.data.size(0);
	};

	std::stable_sort(inputs.begin(), inputs.end(), compCompData);
	std::stable_sort(labels.begin(), labels.end(), compCompData);

	for (int i = 0; i < num; i ++) {
		cout << inputs[i].index << "------------------->" << labels[i].index << endl;
		if (inputs[i].index != labels[i].index) {
			cout << "Failed to match " << endl;
		}
	}
}

static void testPack() {
	std::vector<Tensor> datas;
	std::vector<int> seqLens;
	std::vector<Tensor> labels;
	const int maxLen = 3;
	for (int i = maxLen; i > 0; i --) {
		Tensor data = torch::zeros({i, 4});
		Tensor label = torch::zeros({i, 1});
		datas.push_back(data);
		labels.push_back(label);
		auto dataPtr = data.accessor<float, 2>();
		auto labelPtr = label.accessor<float, 2>();
		for (int j = 0; j < data.size(0); j ++) {
			dataPtr[j][0] = (j + 1) * i;
			labelPtr[j][0] = (j + 1) * i;
		}
		seqLens.push_back(i);
	}
	for (int i = 0; i < maxLen; i ++) {
		datas[i] = torch::constant_pad_nd(datas[i], {0, 0, 0, (maxLen - seqLens[i])});
		labels[i] = torch::constant_pad_nd(labels[i], {0, 0, 0, (maxLen - seqLens[i])});
	}
	torch::Tensor lens = torch::tensor(seqLens);
	Tensor tmpData = torch::stack(datas, 0);
	cout << "Original padded data: " << endl << tmpData << endl;

	const bool batchFirst = true;
	auto packedData = torch::nn::utils::rnn::pack_padded_sequence(torch::stack(datas, 0), lens, batchFirst);
	cout << "packed len: " << endl <<  packedData.batch_sizes() << endl;
	cout << "packed data: " << endl << packedData.data() << endl;
	auto packedLabel = torch::nn::utils::rnn::pack_padded_sequence(torch::stack(labels, 0), lens, batchFirst);
	cout << "packed label len: " << endl <<  packedLabel.batch_sizes() << endl;
	cout << "packed label: " << endl << packedLabel.data() << endl;

	auto unpacked = torch::nn::utils::rnn::pad_packed_sequence(packedData, batchFirst);
	cout << "unpacked data: " << endl << std::get<0>(unpacked) << endl;
	cout << "unpacked lens: " << endl << std::get<1>(unpacked) << endl;
	auto unpackedLabels = std::get<0>(unpacked);

	std::vector<Tensor> outputs;
	for (int i = 0; i < seqLens.size(); i ++) {
		outputs.push_back(unpackedLabels[i].narrow(0, 0, seqLens[i]));
	}
	Tensor output = torch::cat(outputs, 0);
	cout << "Output: " << endl << output << endl;
}

int main(int argc, char** argv) {
	const float lr = atof(argv[2]);
	const int sampleNum = atoi(argv[1]);
	const int seqLen = 27;
//	test2Rows(lr);
//	test5Rows(lr, sampleNum);
//	testShuffle();
//	test5RowThread(lr);

//	getDbCap();
//	getMaxSeqLen();
//	testGRUStructure();


//	test5RowsDbsMask<GRUMaskNet>(lr, sampleNum, seqLen);
//	testDataQ();
	testTrainByQ<GRUMaskNet>(lr, sampleNum, seqLen);
//	testStableSort();

//	testPack();
}
