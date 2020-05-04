/*
 * cnntest.cpp
 *
 *  Created on: Mar 7, 2020
 *      Author: zf
 */

#include <torch/torch.h>
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
#include <filesystem>
#include <thread>
#include <cmath>
#include <time.h>
#include <ctime>
#include <chrono>

#include "nets/lstmnet.h"
#include "nets/fcnet.h"
#include "nets/FixedFcNet.h"
#include "nets/purefcnet.h"
#include "nets/cnnnet.h"

//#include "pytools/plotserver.h"
#include "pytools/batchplotserver.h"
#include "pytools/syncplotserver.h"

using Tensor = torch::Tensor;
using TensorList = torch::TensorList;
using string = std::string;

using std::endl;
using std::cout;

std::ofstream dataFile("/home/zf/workspaces/workspace_cpp/torchpractice/build/errorstats.txt");

static Tensor getLossWeights() {
	std::vector<int> classObs{33572, 23254, 16203, 14673, 12465, 13748, 16262, 22762, 33612, 33797, 23184, 16583, 13655, 12137, 13609, 16392, 23132, 33620, 33956, 23593, 16791, 13935, 12447, 13962, 16763, 23257, 33918, 37661, 40072, 40991, 41101, 36822, 37162, 37324,15749, 21448, 515, 168, 0, 11807, 15260, 55656};
	float totalObs = 0;
	for (int i = 0; i < classObs.size(); i ++) {
		totalObs += classObs[i];
	}

	std::vector<float> weights(classObs.size(), 0);
	for (int i = 0; i < weights.size(); i ++) {
		if (classObs[i] > 0) {
			weights[i] = totalObs / (classObs.size() * classObs[i]);
		}
		if (weights[i] > 2) {
			weights[i] = 0;
		}
	}

	Tensor wTensor = torch::zeros(weights.size());
	for (int i = 0; i < weights.size(); i ++) {
		wTensor[i] = weights[i];
	}
//
//	for (int i = 0; i < weights.size(); i ++) {
//		std::cout << weights[i] << ", " << wTensor[i] << std::endl;
//	}

	return wTensor;
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


//template<typename DbDefs, typename NetType>
//static std::pair<float, float> validCnnOverfit(LmdbSceneReader<DbDefs>& reader, NetType& net, const int dbStartPos) {
////	const int dbStartPos = 1000;
//	const int batchSize = 1024;
//	reader.reset();
//	net.eval();
//	int pos = 0;
//	while (reader.hasNext() && (pos < dbStartPos)) {
//		reader.next();
//		pos ++;
//	}
//
//	std::vector<Tensor> inputs;
//	std::vector<Tensor> labels;
//	std::tie(inputs, labels) = reader.next(batchSize);
//	Tensor labelTensor = torch::cat(labels, 0);
//	Tensor output = net.forward(inputs, 0, false, false);
//
//	auto loss = torch::nll_loss(output, labelTensor).item<float>();
//	auto accu = evaluation(output, labelTensor, false);
//
//	return std::make_pair(loss, accu);
//}

template<typename NetType>
static std::pair<float, float> validCnnOverfit(std::vector<Tensor>& inputs, std::vector<Tensor>& labels, NetType& net, SyncPlotServer& plotServer)
{
	//	const int dbStartPos = 1000;
	net.eval();

	Tensor labelTensor = torch::cat(labels, 0);
	Tensor output = net.forward(inputs, 0, false, false);

	auto loss = torch::nll_loss(output, labelTensor, getLossWeights(), at::Reduction::Mean, 41).item<float>();
	auto accu = evaluation(output, labelTensor, true);

	plotServer.validUpdate(output, labelTensor);

	return std::make_pair(loss, accu);
}

static std::vector<float> updateRatio(std::vector<Tensor>& lastWs, std::vector<Tensor>& ws) {
	std::vector<float> updates;
	for (int i = 0; i < lastWs.size(); i ++) {
		Tensor updateWs = ws[i].sub(lastWs[i]);
		Tensor ratioTensor = updateWs.norm().div(lastWs[i].norm());
		float ratio = ratioTensor.item<float>();
		updates.push_back(log(ratio));
	}

	return updates;
}

template<typename NetType, typename DbDefs>
static void trainCnnDbOverfit(LmdbSceneReader<DbDefs>& reader, NetType& net, torch::optim::Optimizer& optimizer, const int sampleNum) {
//	torch::optim::Adam optimizer (net.parameters(), torch::optim::AdamOptions(0.01));
	const int len = 8;
	optimizer.zero_grad();
	int64_t totalNum = 0;
	double totalLoss = 0;
	const int epoch = 32;
	const int batchSize = 128;
//	const int sampleNum = 8192;
	const int validStep = 128;
	const int earlyStop = 32;
	const float validRatio = 0.001;
	int stopStep = 0;
	float lastLoss = FLT_MAX;
	int count = 0;
//	std::cout << "conv0 weight shape " << 	net.conv0->weight.sizes() << std::endl;
	std::vector<float> iteVec;
	std::vector<float> lossVec;
	std::map<std::string, std::string> keys;


	SyncPlotServer plotServer(net.parameters().size(), net.parameters());

//	std::vector<Tensor> lastWs;
//	for (int i = 0; i < optimizer.parameters().size(); i ++) {
//		lastWs.push_back(optimizer.parameters()[i].clone());
//	}

//	auto& plotServer = BatchPlotServer::GetInstance();
//	std::cout << "Get server instance" << std::endl;
//	PlotServer::Start();
//	std::thread t(BatchPlotServer::Run);
//	std::cout << "Start thread " << std::endl;


	Tensor lossWTensor = getLossWeights();

	std::vector<Tensor> validInputs;
	std::vector<Tensor> validLabels;
	reader.reset();
	for (int i = 0; i < (sampleNum + 100); i ++) {
		reader.next();
	}
	int validSetSize = std::max(1024, (int)(sampleNum * validRatio));
	std::tie(validInputs, validLabels) = reader.next(validSetSize);
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

			std::tie(inputs, labels) = reader.next(batchSize);
//			shuffleInput(inputs, labels);
//			std::cout << "Input " << inputs[0].sizes() << std::endl;
//			std::cout << inputs[0] << std::endl;

			totalNum += inputs.size();


			Tensor output;
			if ((count % validStep) == 0 && !printed) {
				output = net.forward(inputs, len, true, true);
				printed = true;
			} else {
				output = net.forward(inputs, len, true, false);
			}
//			std::cout << "train output " << output.sizes() << std::endl;

//			std::cout << "Labels " << validLabels.size() << std::endl;
//			Tensor labelTensor = at::cat(validLabels, 0);
//			std::cout << "Label tensor: " << labelTensor.sizes() << std::endl;
//			std::cout << "Labels " << labels.size() << std::endl;
			Tensor labelTensor = at::cat(labels, 0);
//			std::cout << "Label tensor: " << labelTensor.sizes() << std::endl;
			//TODO: weighted loss
			auto loss = torch::nll_loss(output, labelTensor, lossWTensor, at::Reduction::Mean, 41);
			totalLoss += (loss.item<float>() * inputs.size());
//			std::cout << "End of loss " << std::endl;
			auto accu = evaluation(output, labelTensor, false);
			std::cout << "====================================================> loss " << loss.item<float>()
				<< ", " << (totalLoss / totalNum)
				<< ", " << accu << std::endl;

//			plotServer.newEvent(std::make_pair(TrainLossIndex, loss.item<float>()));
//			std::cout << "Push event " << std::endl;
//			plotServer.newEvent(std::make_pair(TrainAccuIndex, accu));
//			std::cout << "Push another event" << std::endl;


//			plotStats(loss.item<float>(), accu);
//			iteVec.push_back(iteNum);
//			lossVec.push_back(loss.item<float>());
//			iteNum ++;
//			matplotlibcpp::plot(iteVec, lossVec, keys);
//			matplotlibcpp::pause(0.01);


			loss.backward();
			optimizer.step();


//			std::vector<float> updates = updateRatio(lastWs, optimizer.parameters());
//			cout << "******************************************* Updates ***********" << endl;
//			for (int i = 0; i < updates.size(); i ++) {
//				cout << updates[i] << ", " ;
//			}
//			cout << endl;
//			for (int i = 0; i < updates.size(); i ++) {
//				lastWs[i] = optimizer.parameters()[i].clone();
//			}
			plotServer.trainUpdate(output, labelTensor, net.parameters());
			iteNum = (iteNum + 1) % 65535;
			if (iteNum % 32 == 0) {
				plotServer.refresh();
			}
		}
//		if ((count % validStep) == 0) {
//			printConvWeight(count, net);
//
//			auto validLoss = validOverfit(reader, net, len, (sampleNum + 10));
//			std::cout << "---------------------------------> Validation loss: " << std::get<0>(validLoss)
//					<< ", " << std::get<1>(validLoss) << std::endl;
//		}

//		auto validLoss = validCnnOverfit(reader, net, (sampleNum + 100));
		auto validLoss = validCnnOverfit(validInputs, validLabels, net, plotServer);
		std::cout << "---------------------------------> Validation loss: " << std::get<0>(validLoss)
				<< ", " << std::get<1>(validLoss) << std::endl;

//		plotServer.newEvent(std::make_pair(ValidLossIndex, std::get<0>(validLoss)));
//		plotServer.newEvent(std::make_pair(ValidAccuIndex, std::get<1>(validLoss)));

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
//		writeConvWeight(net.conv0->weight);
	}

	auto saveTime = std::chrono::system_clock::now().time_since_epoch();
	auto saveSecond = std::chrono::duration_cast<std::chrono::seconds>(saveTime).count();
//	time_t saveSecond = time(nullptr);
//	auto a = std::ctime(saveTime);
	plotServer.save(net.GetName() + "_" + (std::to_string(saveSecond)));
	std::string modelPath = net.GetName() + "_" + (std::to_string(saveSecond)) + ".pt";
	torch::serialize::OutputArchive output_archive;
	net.save(output_archive);
	output_archive.save_to(modelPath);
//	plotServer.notifyRead();
//	plotServer.stop();
//	t.join();
//	std::cout << "conv0 weights " << std::endl;
//	std::cout << net.conv0->weight << std::endl;

//	printConvWeight(epoch, net);
}


static void test5Rows(const float lr, const int sampleNum) {
	//	dataFile.open("./conv0output.txt");
//		const std::string cppCreateDb = "/home/zf/workspaces/res/dbs/lmdbscenetest";
	const std::string dbPath = "/home/zf/workspaces/res/dbs/lmdbcpptest";
		LmdbSceneReader<LmdbDataDefs> reader(dbPath);
		std::cout << "End of reader construction " << std::endl;
//		LstmNet net;
//		FcNet net;
//		FixedFcNet net;
		CNNNet net;
//		PureFcNet net;
//		torch::optim::SGD optimizer (net.parameters(), torch::optim::SGDOptions(lr));
//		torch::optim::Adam optimizer (net.parameters(), torch::optim::AdamOptions(lr)); //.weight_decay(0.01));
		torch::optim::Adagrad optimizer(net.parameters(), torch::optim::AdagradOptions(lr));
//		torch::optim::RMSprop optimizer(net.parameters(), torch::optim::RMSpropOptions(lr)); //Explore
//		optimizer.square_average_buffers;

	//	train(reader, net);
	//	std::cout << "------------------------------------------------------------------->" << std::endl;
	//	train(reader, net, 8, optimizer);
	//	testPad();
	//	getAveLen(reader, net);
	//	testDb(reader);

	//	trainSameLen(reader, net, 8, optimizer);
		std::cout << "param num " << net.parameters().size() << endl;
//		std::cout << "2: " << net.parameters()[2].sizes() << endl;
//		cout << "6: " << net.parameters()[6].sizes() << endl;
//		cout << "10: " << net.parameters()[10].sizes() << endl;
//		cout << "11: " << net.parameters()[11].sizes() << endl;
		trainCnnDbOverfit(reader, net, optimizer, sampleNum);
}

int main(int argc, char** argv) {
	const float lr = atof(argv[2]);
	const int sampleNum = atoi(argv[1]);
//	test2Rows(lr);
	test5Rows(lr, sampleNum);
//	testShuffle();
//	test5RowThread(lr);
//	getLossWeights();
}
