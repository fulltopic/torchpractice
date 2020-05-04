/*
 * lstmtest.cpp
 *
 *  Created on: Mar 24, 2020
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

#include "pytools/syncplotserver.h"

using Tensor = torch::Tensor;
using TensorList = torch::TensorList;
using string = std::string;

using std::endl;
using std::cout;

const std::string wsPath = "/home/zf/";
std::ofstream dataFile(wsPath + "/workspaces/workspace_cpp/torchpractice/build/errorstats.txt");

std::function<void(int)> shutdownHandler;
void signalHandler(int signal) {
	std::cout << "Received signal " << signal << std::endl;
	shutdownHandler(signal);
	exit(signal);
}

static float evaluation(Tensor outputs, Tensor labels, bool isTest) {
	const int batchSize = outputs.size(0);
	Tensor values;
	Tensor indices;
	indices = torch::argmax(outputs, 1);

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
	for (int i = 0; i < total; i ++) {
		if (labelData[i] != indexData[i]){
			errCounts[(int)labelData[i]][(int)indexData[i]] ++;
			labelErrs[(int)labelData[i]] ++;
		}
	}
	dataFile << "------------------------------------------------------------> Evaluation " << std::endl;
	for (int i = 0; i < errCounts.size(); i ++) {
		dataFile << labelErrs[i] << ": ";
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

//	plotServer.validUpdate(output, labelTensor);

	return std::make_pair(loss, accu);
}

template<typename NetType>
static std::pair<float, float> validLstmOverfitWithPlot(std::vector<Tensor>& inputs, std::vector<Tensor>& labels, const int seqLen, NetType& net, SyncPlotServer& plotServer)
{
	net.eval();

	Tensor labelTensor = torch::cat(labels, 0);
	Tensor output = net.forward(inputs, seqLen, false, false);

	auto loss = torch::nll_loss(output, labelTensor).item<float>();
	auto accu = evaluation(output, labelTensor, true);

	plotServer.validUpdate(output, labelTensor);
	plotServer.refresh();

	return std::make_pair(loss, accu);
}

template<typename NetType, typename PlotServerType>
static void saveOutput(NetType& net, PlotServerType& plotServer) {
	auto saveTime = std::chrono::system_clock::now().time_since_epoch();
	auto saveSecond = std::chrono::duration_cast<std::chrono::seconds>(saveTime).count();
	plotServer.save(net.GetName() + "_" + (std::to_string(saveSecond)));
	std::string modelPath = net.GetName() + "_" + (std::to_string(saveSecond)) + ".pt";
	torch::serialize::OutputArchive output_archive;
	net.save(output_archive);
	output_archive.save_to(modelPath);
}

template<typename NetType, typename DbDefs>
static void trainLstmDbOverfit(LmdbSceneReader<DbDefs>& reader, NetType& net,
		torch::optim::Optimizer& optimizer, const int sampleNum, const int seqLen) {

	optimizer.zero_grad();
	int64_t totalNum = 0;
	double totalLoss = 0;
	const int epoch = 32;
	const int batchSize = 32;
	const int validStep = 128;
	const int earlyStop = 32;
	const float validRatio = 0.001;
	int stopStep = 0;
	float lastLoss = FLT_MAX;
	int count = 0;
	std::vector<float> iteVec;
	std::vector<float> lossVec;
	std::map<std::string, std::string> keys;

	SyncPlotServer plotServer(net.parameters().size(), net.parameters(), "Sample# " + std::to_string(sampleNum));

	shutdownHandler = [&](int signal) {
		saveOutput(net, plotServer);
	};
	std::signal(SIGABRT, signalHandler);
	std::signal(SIGFPE, signalHandler);
	std::signal(SIGILL, signalHandler);
	std::signal(SIGINT, signalHandler);
	std::signal(SIGSEGV, signalHandler);
	std::signal(SIGTERM, signalHandler);

	std::vector<Tensor> validInputs;
	std::vector<Tensor> validLabels;
	reader.reset();
	reader.next(sampleNum, seqLen);
	reader.next();
//	for (int i = 0; i < (sampleNum * 1.5); i ++) {
//		reader.next();
//	}
	int validSetSize = std::max(256, (int)(sampleNum * validRatio));
	std::tie(validInputs, validLabels) = reader.next(validSetSize, seqLen);
	cout << "Get validation dataset: " << validInputs.size() << endl;
	reader.reset();


	int iteNum = 1;
	while (count <= epoch) {
		totalNum = 0;
		totalLoss = 0;
		bool printed = false;
		reader.reset();
		net.train();
		while (totalNum < sampleNum) {
			std::vector<Tensor> inputs;
			std::vector<Tensor> labels;

			std::tie(inputs, labels) = reader.next(batchSize, seqLen);

			totalNum += inputs.size();


			Tensor output;
			if ((count % validStep) == 0 && !printed) {
				output = net.forward(inputs, seqLen, true, true);
				printed = true;
			} else {
				output = net.forward(inputs, seqLen, true, false);
			}

			Tensor labelTensor = at::cat(labels, 0);
			auto loss = torch::nll_loss(output, labelTensor);
			totalLoss += (loss.item<float>() * inputs.size());
			auto accu = evaluation(output, labelTensor, false);
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
		}

		auto validLoss = validLstmOverfitWithPlot(validInputs, validLabels, seqLen, net, plotServer);
		std::cout << "---------------------------------> Validation loss: " << std::get<0>(validLoss)
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
		count ++;
	}

	saveOutput(net, plotServer);

}


template<typename NetType, typename DbDefs>
static void trainLstmDbOverfit(std::vector<LmdbSceneReader<DbDefs>>& readers,
		LmdbSceneReader<DbDefs>& validReader,
		NetType& net, torch::optim::Optimizer& optimizer, const int sampleNum, const int seqLen) {

	optimizer.zero_grad();
	int64_t totalNum = 0;
	double totalLoss = 0;
	const int epoch = 1;
	const int batchSize = 32;
	const int validStep = 128;
	const int earlyStop = 32;
	const float validRatio = 0.001;
	int stopStep = 0;
	float lastLoss = FLT_MAX;
//	int count = 0;
	std::vector<float> iteVec;
	std::vector<float> lossVec;
	std::map<std::string, std::string> keys;

	SyncPlotServer plotServer(net.parameters().size(), net.parameters(), "Sample# " + std::to_string(sampleNum));

	shutdownHandler = [&](int signal) {
		saveOutput(net, plotServer);
	};
	std::signal(SIGABRT, signalHandler);
	std::signal(SIGFPE, signalHandler);
	std::signal(SIGILL, signalHandler);
	std::signal(SIGINT, signalHandler);
	std::signal(SIGSEGV, signalHandler);
	std::signal(SIGTERM, signalHandler);

	std::vector<Tensor> validInputs;
	std::vector<Tensor> validLabels;

	int validSetSize = std::max(256, (int)(sampleNum * validRatio));
	std::tie(validInputs, validLabels) = validReader.next(validSetSize, seqLen);
	cout << "Get validation dataset: " << validInputs.size() << endl;
	validReader.reset();


	int iteNum = 1;

	for (int count = 0; count <= epoch; count ++) {
		totalNum = 0;
		totalLoss = 0;
		bool printed = false;

		int readerIndex = 0;
		for (int i = 0; i < readers.size(); i ++) {
			readers[i].reset();
		}

		net.train();
		while (totalNum < sampleNum) {
			std::vector<Tensor> inputs;
			std::vector<Tensor> labels;

			if (!readers[readerIndex].hasNext()) {
				readerIndex ++;
				if (readerIndex >= readers.size()) {
					break;
				}
			}
			std::tie(inputs, labels) = readers[readerIndex].next(batchSize, seqLen);

			totalNum += inputs.size();


			Tensor output = net.forward(inputs, seqLen, true, false);

			Tensor labelTensor = at::cat(labels, 0);
			auto loss = torch::nll_loss(output, labelTensor);
			totalLoss += (loss.item<float>() * inputs.size());
			auto accu = evaluation(output, labelTensor, false);
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
		}

		auto validLoss = validLstmOverfitWithPlot(validInputs, validLabels, seqLen, net, plotServer);
		std::cout << "---------------------------------> Validation loss: " << std::get<0>(validLoss)
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

	saveOutput(net, plotServer);

}

template<typename NetType>
static void test5Rows(const float lr, const int sampleNum, const int seqLen) {
//	const std::string dbPath = "/home/zf/workspaces/res/dbs/lmdbscenetest";
	const std::string dbPath =  wsPath + "/workspaces/res/dbs/lmdb5rowscenetest";
	LmdbSceneReader<LmdbDataDefs> reader(dbPath);
	std::cout << "End of reader construction " << std::endl;
	NetType net(seqLen);
//	GRUNet net(seqLen);

//	torch::optim::SGD optimizer (net.parameters(), torch::optim::SGDOptions(lr));
//	torch::optim::Adam optimizer (net.parameters(), torch::optim::AdamOptions(lr)); //.weight_decay(0.01));
	torch::optim::Adagrad optimizer(net.parameters(), torch::optim::AdagradOptions(lr));
//	torch::optim::RMSprop optimizer(net.parameters(), torch::optim::RMSpropOptions(lr)); //Explore
//	optimizer.square_average_buffers;

	std::cout << "param num " << net.parameters().size() << endl;
	trainLstmDbOverfit(reader, net, optimizer, sampleNum, seqLen);
}

template<typename NetType>
static void test5RowsVecDbs(const float lr, const int sampleNum, const int seqLen) {
	std::vector<std::string> dbPaths {
		wsPath + "/workspaces/res/dbs/lmdb5rowscenetest",
		wsPath + "/workspaces/res/dbs/lmdb5rowscenetest_3",
		wsPath + "/workspaces/res/dbs/lmdb5rowscenetest_4"
	};

	std::vector<LmdbSceneReader<LmdbDataDefs>> readers (dbPaths.begin(), dbPaths.end());

	std::string validDbPath = wsPath + "/workspaces/res/dbs/lmdb5rowscenetestvalid";
	LmdbSceneReader<LmdbDataDefs> validReader(validDbPath);

	NetType net(seqLen);
	torch::optim::Adagrad optimizer(net.parameters(), torch::optim::AdagradOptions(lr));

	trainLstmDbOverfit(readers, validReader, net, optimizer, sampleNum, seqLen);
}

//Currently, model seqlen = 10
template<typename NetType>
static void testLoadedModel(const std::string modelName, const int sampleNum, const int seqLen) {
	std::string dbPath = wsPath + "/workspaces/res/dbs/testdatasetlmdb5rowscenetest";
//	std::string modelName = "GRUNet_1585591364.pt";
	std::string modelPath = wsPath + "/workspaces/workspace_cpp/torchpractice/build/models/" + modelName;
	std::cout << "Model path " << modelPath << std::endl;

//	std::ifstream modelIn(modelPath);
//	torch::jit::script::Module model = torch::jit::load(modelIn, c10::DeviceType::CPU);
//


	NetType net(seqLen);

	torch::serialize::InputArchive inChive;
	inChive.load_from(modelPath);
	net.load(inChive);
//
//	torch::OrderedDict<std::string, torch::Tensor>  params = net.named_parameters(true);
//	for (auto ite = params.begin(); ite != params.end(); ite ++) {
//		std::cout << ite->key() << ": " << ite->value().sizes() << std::endl;
//
//		if (ite->key().compare("lstm0.bias_ih_l0") == 0) {
//			std::cout << ite->value().data_ptr<float>()[0] << std::endl;
//		}
//	}
//
	net.eval();
	LmdbSceneReader<LmdbDataDefs> reader(dbPath);
	std::vector<torch::Tensor> testInputs;
	std::vector<torch::Tensor> testLabels;
	std::tie(testInputs, testLabels) = reader.next(sampleNum, seqLen);

	auto rc = testLstmOverfit(testInputs, testLabels, seqLen, net);
	std::cout << "Result: " << rc.first << ", " << rc.second << std::endl;
}

static void testGRUStructure () {
	GRUNet net(10);

//	auto buffers = net.named_buffers(true);
//	for (auto ite = buffers.begin(); ite != buffers.end(); ite ++) {
//		std::cout << ite->key() << ": " << ite->value().sizes() << std::endl;
//	}
//
//	auto params = net.named_parameters(true);
//	for (auto ite = params.begin(); ite != params.end(); ite ++) {
//		std::cout << ite->key() << ": " << ite->value().sizes() << std::endl;
//
//		if (ite->key().compare("lstm0.bias_ih_l0") == 0) {
//			std::cout << ite->value().data_ptr<float>()[0] << std::endl;
//			ite->value().data<float>()[0] = 1;
//		}
//	}



//	std::cout << "Print bias structure " << std::endl;
//	for (auto ite = params.begin(); ite != params.end(); ite ++) {
//		std::cout << "Get key " << ite->key() << std::endl;
//		if ((ite->key().compare("lstm0.bias_ih_l0") == 0) || (ite->key().compare("lstm0.bias_hh_l0") == 0)) {
//			auto dataPtr = ite->value().data_ptr<float>();
//			std::cout << "samples before: " << dataPtr[0] << ", " << dataPtr[100] << ", " << dataPtr[1000] << std::endl;
//			std::cout << ite->value().sizes() << std::endl;
//
//			auto chunks = ite->value().chunk(3, 0);
//			chunks[0].fill_(-1);
//
//			std::cout << "samples after: " << dataPtr[0] << ", " << dataPtr[100] << ", " << dataPtr[1000] << std::endl;
//		}
//	}
}

static void getDbCap() {
	const std::string dbPath = wsPath + "/workspaces/res/dbs/lmdb5rowscenetest";
	LmdbSceneReader<LmdbDataDefs> reader(dbPath);
	int count = 0;

	while (reader.hasNext()) {
		reader.next();
		count ++;
	}
	std::cout << "Db cap: " << count << std::endl;
}

int main(int argc, char** argv) {
	const float lr = atof(argv[2]);
	const int sampleNum = atoi(argv[1]);
	const int seqLen = 10;
//	test2Rows(lr);
//	test5Rows(lr, sampleNum);
//	testShuffle();
//	test5RowThread(lr);

//	getDbCap();

//	testLoadedModel<GRUNet>("GRUNet_1585855488.pt", sampleNum, seqLen);
//	testLoadedModel<LstmNet>("./LstmNet_1585131414.pt", sampleNum, seqLen);
//	test5Rows<LstmNet>(lr, sampleNum, seqLen);
	test5RowsVecDbs<LstmNet>(lr, sampleNum, seqLen);
//	test5Rows<GRUNet>(lr, sampleNum, seqLen);
//	test5Rows<FcGRUNet>(lr, sampleNum, seqLen);
//	test5RowsVecDbs<GRUNet>(lr, sampleNum, seqLen);
//	test5RowsVecDbs<CnnGRUNet>(lr, sampleNum, seqLen);
//	test5RowsVecDbs<FcGRUNet>(lr, sampleNum, seqLen);
//	test5RowsVecDbs<Lstm2Net>(lr, sampleNum, seqLen);
//	test5RowsVecDbs<LstmNet>(lr, sampleNum, seqLen);

//	testGRUStructure();
}
