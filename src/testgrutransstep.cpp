/*
 * testgrutransstep.cpp
 *
 *  Created on: May 20, 2020
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

#include "pytools/syncplotserver.h"

#include "nets/supervisednet/grutransstep.h"

using torch::Tensor;
using torch::TensorList;

using std::string;
using std::vector;
using std::endl;
using std::cout;

const string wsPath = "/home/zf/";
const int batchSize = 32;

std::function<void(int)> shutdownHandler;
void signalHandler(int signal) {
	std::cout << "Received signal " << signal << std::endl;
	shutdownHandler(signal);
	exit(signal);
}


template<typename NetType, typename PlotServerType>
static void saveOutput(NetType& net, PlotServerType& plotServer, const int epochCount) {
	auto saveTime = std::chrono::system_clock::now().time_since_epoch();
	auto saveSecond = std::chrono::duration_cast<std::chrono::seconds>(saveTime).count();
	plotServer.save(net.GetName() + "_" + std::to_string(epochCount) + "_" + (std::to_string(saveSecond)));
	std::string modelPath = net.GetName() + "_" + (std::to_string(saveSecond)) + ".pt";
	torch::serialize::OutputArchive output_archive;
	net.save(output_archive);
	output_archive.save_to(modelPath);
}


static float evaluation(Tensor outputs, Tensor labels, bool isTest) {
	const int batchSize = outputs.size(0);
	Tensor values;
	Tensor indices;
	indices = torch::argmax(outputs, 1);

	Tensor diff = torch::sub(labels, indices);
	int total = labels.size(0);
	auto matched = total - diff.nonzero().size(0);

	const long* labelData = labels.data_ptr<long>();
	const long* indexData = indices.data_ptr<long>();

	if (isTest) {
		std::vector<std::vector<int>> errCounts(42, std::vector<int>(42, 0));
		std::vector<int> labelErrs(42, 0);
		for (int i = 0; i < total; i ++) {
			if (labelData[i] != indexData[i]){
				errCounts[(int)labelData[i]][(int)indexData[i]] ++;
				labelErrs[(int)labelData[i]] ++;
			}
		}
	}

	float accu = (float)matched / total;

	return accu;
}

//Return {batch, seq, 360}
static Tensor mergeTensor(vector<Tensor>& inputs, const int seqLen) {
	for (int i = 0; i < inputs.size(); i ++) {
		inputs[i] = inputs[i].view({1, seqLen, 360});
	}
	return torch::cat(inputs, 0);
}

template<typename NetType>
static std::pair<float, float> validLstmOverfitWithPlot(std::vector<Tensor>& inputs, std::vector<Tensor>& labels, const int seqLen, NetType& net, SyncPlotServer& plotServer)
{
	cout << "================================> validation: " << inputs.size() << ", " << labels.size() << endl;
	net.eval();

	vector<Tensor> outputs;
	Tensor output;

	//{batchSize * seqLen, 5, 72}
	Tensor inputSet = torch::cat(inputs, 0);
	//{batchSize * seqLen, 360}
	inputSet = inputSet.view({inputSet.size(0), inputSet.size(1) * inputSet.size(2)});
	//{batchSize, seqLen, 360}
	inputSet = inputSet.view({inputSet.size(0) / seqLen, seqLen, inputSet.size(1)});
	std::cout << "inputSet " << inputSet.sizes() << std::endl;
	vector<Tensor> inputVec = inputSet.chunk(seqLen, 1);
	cout << "inputVec: " << inputVec.size() << " input: " << inputVec[0].sizes() << endl;

	for (int i = 0; i < seqLen; i ++) {
		auto output = net.forward({inputVec[i].div(4)}, false);
		outputs.push_back(output[0]);
	}

	//{batchSize, 42}
	cout << outputs.size() << ": " << outputs[0].sizes() << endl;
	//{batchSize, seqLen * 42}
	output = torch::cat(outputs, 1);
	cout << "output " << output.sizes() << endl;
	output = output.view({output.size(0), seqLen, output.size(1) / seqLen});
	cout << "output " << output.sizes() << endl;
	output = output.view({output.size(0) * output.size(1), output.size(2)});

	Tensor labelTensor = torch::cat(labels, 0);

	auto loss = torch::nll_loss(output, labelTensor).item<float>();
	auto accu = evaluation(output, labelTensor, true);

	return {loss, accu};

//	return {0.0f, 0.0f};
}

#define bindSigHandles(handler) {	\
	std::signal(SIGABRT, handler); \
	std::signal(SIGFPE, handler); \
	std::signal(SIGILL, handler); \
	std::signal(SIGINT, handler); \
	std::signal(SIGSEGV, handler); \
	std::signal(SIGTERM, handler); \
}

template<typename NetType, typename DbDefs>
static void trainLstmDbOverfit(LmdbSceneReader<DbDefs>& reader, NetType& net,
		torch::optim::Optimizer& optimizer, const int sampleNum, const int seqLen) {

	optimizer.zero_grad();
	int64_t totalNum = 0;
	double totalLoss = 0;
	const int epoch = 64;
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
		saveOutput(net, plotServer, 1024);
	};
	bindSigHandles(signalHandler);


	std::vector<Tensor> validInputs;
	std::vector<Tensor> validLabels;
	reader.reset();
	reader.next(sampleNum, seqLen);
	reader.next();

	int validSetSize = std::max(256, (int)(sampleNum * validRatio));
	std::tie(validInputs, validLabels) = reader.next(validSetSize, seqLen);
	cout << "Get validation dataset: " << validInputs.size() << endl;
	reader.reset();
	cout << "============================> Training begin " << endl;

	int iteNum = 1;
	while (count <= epoch) {
		totalNum = 0;
		totalLoss = 0;
		bool printed = false;
		reader.reset();
		net.train();
		cout << "--------------------------------------> epoch loop " << totalNum << ", " << sampleNum << endl;
		while (totalNum < sampleNum) {
//			cout << "--------------------------> Training loop " << endl;
			std::vector<Tensor> inputs;
			std::vector<Tensor> labels;

			std::tie(inputs, labels) = reader.next(batchSize, seqLen);

			totalNum += inputs.size();


			vector<Tensor> outputs;
			net.reset();
			Tensor netInput = mergeTensor(inputs, seqLen);
//			cout << "Raw input sizes: " << netInput.sizes() << endl;
			auto netInputs = netInput.chunk(seqLen, 1);
//			cout << "Input sizes: " << netInputs[0].sizes() << endl;

			for (auto stepInput: netInputs) {
//				stepInput = stepInput.view({batchSize, 1, 5 * 72});
				Tensor output = net.forward({stepInput}, true);
//				output = torch::softmax(output, 2);
				//{batch, 1, 42}
				cout << "Output sizes: " << output.sizes() << endl;
				outputs.push_back(output);
			}

			Tensor output = torch::cat(outputs, 1);
//			cout << "output sizes " << output.sizes() << endl;
			output = output.view({batchSize * seqLen, 42});
//			cout << "output sizes " << output.sizes() << endl;
			Tensor labelTensor = at::cat(labels, 0);
//			cout << "label sizes " << labelTensor.sizes() << endl;
			auto loss = torch::nll_loss(output, labelTensor);
//			cout << "Get loss " << loss << endl;
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
			saveOutput(net, plotServer, count);
		} else {
			stopStep ++;
			if (stopStep > earlyStop) {
				std::cout << "Early stop " << std::endl;
				break;
			}
		}
		count ++;
	}

	saveOutput(net, plotServer, epoch);

}


template<typename NetType>
static void test5Rows(const float lr, const int sampleNum, const int seqLen) {
	const std::string dbPath = "/home/zf/workspaces/res/dbs/lmdbscenetest";
//	const std::string dbPath =  wsPath + "/workspaces/res/dbs/lmdb5rowscenetest";
	LmdbSceneReader<LmdbDataDefs> reader(dbPath);
	std::cout << "End of reader construction " << std::endl;
	NetType net;
//	GRUNet net(seqLen);

//	torch::optim::SGD optimizer (net.parameters(), torch::optim::SGDOptions(lr));
//	torch::optim::Adam optimizer (net.parameters(), torch::optim::AdamOptions(lr)); //.weight_decay(0.01));
	torch::optim::Adagrad optimizer(net.parameters(), torch::optim::AdagradOptions(lr));
//	torch::optim::RMSprop optimizer(net.parameters(), torch::optim::RMSpropOptions(lr)); //Explore
//	optimizer.square_average_buffers;

	std::cout << "param num " << net.parameters().size() << endl;
	trainLstmDbOverfit(reader, net, optimizer, sampleNum, seqLen);
}

static void testTransStep(const float lr, const int sampleNum, const int seqLen) {
	const std::string dbPath = "/home/zf/workspaces/res/dbs/lmdbscenetest";
	LmdbSceneReader<LmdbDataDefs> reader(dbPath);

	std::string modelPath = "/home/zf/workspaces/workspace_cpp/torchpractice/build/GRUNet_1589879376.pt";
//	std::string modelPath = "/home/zf/workspaces/workspace_cpp/torchpractice/build/models/candidates/GRUNet_1585782523.pt";

	GRUTransStepNet net;
	net.loadParams(modelPath);

	SyncPlotServer plotServer(net.parameters().size(), net.parameters(), "Sample# " + std::to_string(sampleNum));

//	torch::optim::Adagrad optimizer(net.parameters(), torch::optim::AdagradOptions(lr));
//	std::cout << "param num " << net.parameters().size() << endl;
//	trainLstmDbOverfit(reader, net, optimizer, sampleNum, seqLen);

	std::vector<Tensor> inputs;
	std::vector<Tensor> labels;
	std::tie(inputs, labels) = reader.next(sampleNum, 10);

	auto rc = validLstmOverfitWithPlot<GRUTransStepNet>(inputs, labels, 10, net, plotServer);
	cout << std::get<0>(rc) << ", " << std::get<1>(rc) << endl;
}

int main(int argc, char** argv) {
	const int sampleNum = atoi(argv[1]);
	const float lr = atof(argv[2]);
	const int seqLen = 10;

//	test5Rows<GRUStepNet>(lr, sampleNum, seqLen);
	testTransStep(lr, sampleNum, 10);
}


