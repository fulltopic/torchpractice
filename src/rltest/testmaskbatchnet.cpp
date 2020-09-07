/*
 * testmaskbatchnet.cpp
 *
 *  Created on: Sep 3, 2020
 *      Author: zf
 */

#include <vector>
#include <string>
#include <iostream>

#include <torch/torch.h>

#include "lmdbtools/LmdbReaderWrapper.h"
#include "lmdbtools/LmdbReader.h"
#include "lmdbtools/LmdbDataDefs.h"
#include "lmdbtools/Lmdb2RowDataDefs.h"

#include "rltest/maskbatchnet.h"


using Tensor = torch::Tensor;
using std::cout;
using std::endl;
using std::vector;
using rltest::GRUMaskNet;

const std::string wsPath = "/home/zf/";
std::ofstream dataFile(wsPath + "/workspaces/workspace_cpp/torchpractice/build/errorstats.txt");
const std::string modelPath = "/home/zf/workspaces/workspace_cpp/aws/GRUMaskBatch2048Net_140000_0.002000_1594188556.pt";
const int batchSize = 128;


static bool compTensorBySeqLen (const Tensor& t0, const Tensor& t1) {
	return t0.size(0) > t1.size(0);
}

static void processActionInput(Tensor& input, Tensor& label) {
	auto inputPtr = input.accessor<float, 3>();
	auto labelPtr = label.data_ptr<long>();
	for (int j = 0; j < label.size(0); j ++) {
		if (labelPtr[j] >= 34) {
//				cout << "Label is -------------------------> " << labelPtr[j] << endl;
			for (int k = 34; k < 42; k ++) {
				inputPtr[j][1][k] = 1;
			}
		}
	}
}

static void processActionInput(std::vector<Tensor>& inputs, std::vector<Tensor>& labels) {
	cout << inputs.size() << std::endl;
	cout << labels.size() << std::endl;
	cout << "labels sizes: " << labels[0].sizes() << endl;
	int totalInput = 0;
	int totalLabel = 0;
	for (int i = 0; i < inputs.size(); i ++) {
		totalInput += inputs[i].size(0);
		totalLabel += labels[i].size(0);

		processActionInput(inputs[i], labels[i]);
		inputs[i] = inputs[i].div(4);
//		auto inputPtr = inputs[i].accessor<float, 3>();
//		auto labelPtr = labels[i].data_ptr<long>();
//		for (int j = 0; j < labels[i].size(0); j ++) {
//			if (labelPtr[j] >= 34) {
////				cout << "Label is -------------------------> " << labelPtr[j] << endl;
//				for (int k = 34; k < 42; k ++) {
//					inputPtr[j][1][k] = 1;
//				}
////				cout << "Update input: " << inputs[i][j] << endl;
//			}
//		}
	}
	cout << "-----------------------> processActionInput: " << totalInput << ", " << totalLabel << endl;
	std::stable_sort(inputs.begin(), inputs.end(), compTensorBySeqLen);
	std::stable_sort(labels.begin(), labels.end(), compTensorBySeqLen);
}

static Tensor createLabelTensor(std::vector<Tensor>& labels, const int seqLen) {
	int count = 0;
	int before = 0;
	cout << "seqLen = " << seqLen << endl;
	for (int i = 0; i < labels.size(); i ++) {
		before += labels[i].size(0);
		if (labels[i].size(0) > seqLen) {
			std::cout << "narrowed: " << labels[i].size(0) << std::endl;
			labels[i] = labels[i].narrow(0, 0, seqLen);
		} else {
//			std::cout << "remain: " << labels[i].size(0) << std::endl;
		}
		count += (int)labels[i].size(0);
	}
	cout << "before " << before << endl;
	std::cout << "Total ------------------------------> " << count << std::endl;

	return torch::cat(labels, 0);
}

static void evalAccu(Tensor output, Tensor labelTensor) {
	Tensor indices = torch::argmax(output, 1);
	Tensor diff = torch::sub(labelTensor, indices);

	int total = labelTensor.size(0);
	auto matched = total - diff.nonzero().size(0);
	float accu = (float)matched / total;

	cout << "accu: " << labelTensor.size(0) << ", " << accu << endl;
	//TODO: print out result
}

static void evalModel(GRUMaskNet& net, LmdbSceneReader<LmdbDataDefs>& validReader, const int sampleNum, const int seqLen) {
	cout << sampleNum << ", " << seqLen << endl;
	validReader.reset();
	vector<Tensor> validInputs;
	vector<Tensor> validLabels;
	std::tie(validInputs, validLabels) = validReader.next(sampleNum);
	cout << "samples: " << validInputs.size() << endl;
	processActionInput(validInputs, validLabels);

	Tensor labelTensor = createLabelTensor(validLabels, seqLen);

	net.eval();
	auto output = net.forward(validInputs, true);

	auto loss = torch::nll_loss(output[0], labelTensor).item<float>();
	evalAccu(output[0], labelTensor);

}

static void evalStepModel(GRUMaskNet& net, LmdbSceneReader<LmdbDataDefs>& validReader, const int sampleNum) {
	validReader.reset();
	vector<Tensor> validInputs;
	vector<Tensor> validLabels;
	std::tie(validInputs, validLabels) = validReader.next(sampleNum);

	for (int sample = 0; sample < validInputs.size(); sample ++) {
		Tensor input = validInputs[sample];
		Tensor label = validLabels[sample];
		processActionInput(input, label);

		net.eval();
		const int seqLen = input.size(0);
//		cout << "input sizes " << input.sizes() << endl;
//		cout << "seqLen: " << seqLen << endl;

		vector<Tensor> outputs;
		outputs.reserve(seqLen);
		Tensor hState = net.createHState();
		for (int i = 0; i < seqLen; i ++) {
			Tensor netInput = input[i].view({1, 1, 360}).div(4);
			auto output = net.forward({netInput, hState, torch::tensor(i)});
			hState = output[1];
			outputs.push_back(output[0].view({1, 42}));
		}

//		cout << "output sizes " << outputs[0].sizes() << endl;
		Tensor outTensor = torch::cat(outputs, 0);
//		cout << "outputTensor " << outTensor << endl;
//		cout << "label " << label.sizes() << endl;

		evalAccu(outTensor, label);
	}
}

static void testTrain(const int sampleNum, const int seqLen, const int type) {
	std::string validDbPath = wsPath + "/workspaces/res/dbs/lmdb5rowscenetestvalid";
	cout << "validDbPath: " << validDbPath << endl;
	LmdbSceneReader<LmdbDataDefs> validReader(validDbPath);

	GRUMaskNet net(seqLen);
	net.loadModel(modelPath);
	cout << "Model loaded " << endl;

	if (type == 0) {
		evalModel(net, validReader, sampleNum, seqLen);
	} else {
		evalStepModel(net, validReader, sampleNum);
	}
}


int main(int argc, char** argv) {
	const int type = atoi(argv[1]);
	const int sampleNum = atoi(argv[2]);
	const int seqLen = 27;

	testTrain(sampleNum, seqLen, type);

//	testPack();
}
