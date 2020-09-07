/*
 * loadmodeltest.cpp
 *
 *  Created on: Aug 30, 2020
 *      Author: zf
 */

#include <string.h>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <stdexcept>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <torch/torch.h>

#include <cpprl/cpprl.h>
#include <boost/lockfree/queue.hpp>
#include <matplotlibcpp.h>

#include "gymtest/communicator.h"
#include "gymtest/requests.h"
#include "gymtest/gympolicy.h"
#include "gymtest/meanstd.h"
#include "gymtest/gymnetutils.h"
#include "pytools/syncplotserver.h"
#include "lmdbtools/LmdbReaderWrapper.h"
#include "lmdbtools/LmdbReader.h"
#include "lmdbtools/LmdbDataDefs.h"
#include "lmdbtools/Lmdb2RowDataDefs.h"



using Tensor = torch::Tensor;
using string = std::string;
using std::cout;
using std::endl;
using TensorList = torch::TensorList;
using std::vector;


namespace {
//using DbDataType = std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>>;

struct GRUMaskNet: torch::nn::Module {
private:
//	const unsigned int numInputs;
//	const unsigned int numActOutput;
	torch::nn::GRU gru0;
	torch::nn::Linear fc;

	const int seqLen;

public:
	void initParams() {
		auto params = this->named_parameters(true);
		for (auto ite = params.begin(); ite != params.end(); ite ++) {
			std::cout << "Get key " << ite->key() << std::endl;
	//		if ((ite->key().compare("gru0.bias_ih_l0") == 0) || (ite->key().compare("gru0.bias_hh_l0") == 0)) {
			if ((ite->key().find("gru") != std::string::npos) && (ite->key().find("bias") != std::string::npos)) {
				auto dataPtr = ite->value().data_ptr<float>();
				std::cout << "bias samples before: " << dataPtr[0] << ", " << dataPtr[100] << ", " << dataPtr[1000] << std::endl;
				std::cout << ite->value().sizes() << std::endl;

				auto chunks = ite->value().chunk(3, 0);
				chunks[0].fill_(-1);

				std::cout << "bias samples after: " << dataPtr[0] << ", " << dataPtr[100] << ", " << dataPtr[1000] << std::endl;
			}

			if ((ite->key().find("gru") != std::string::npos) && (ite->key().find("weight") != std::string::npos)) {
				auto sizes = ite->value().sizes();
				int dataSize = 1;
				for (int i = 0; i < sizes.size(); i ++) {
					dataSize *= sizes[i];
				}
				int chunkSize = dataSize / 3;

				auto chunks = ite->value().chunk(3, 0);
				auto dataPtr = ite->value().data_ptr<float>();
				std::cout << "weights samples before: " << dataPtr[0] << ", " << dataPtr[chunkSize]
					<< ", " << dataPtr[chunkSize * 2] << std::endl;


				for (int i = 0; i < 3; i ++) {
					auto dataTensor = torch::randn(chunks[i].sizes());
					auto meanTensor = torch::mean(dataTensor);
					std::cout << "Mean " << meanTensor.item<float>() << std::endl;
					auto varTensor = torch::var(dataTensor);
					std::cout << "Var " << varTensor.item<float>() << std::endl;
					dataTensor = dataTensor.div(sqrt((double)chunkSize));

					auto rndPtr = dataTensor.data_ptr<float>();
					auto chunkPtr = chunks[i].data_ptr<float>();
					for (int j = 0; j < chunkSize; j ++) {
						chunkPtr[j] = rndPtr[j];
					}
				}

				std::cout << "weights samples after: " << dataPtr[0] << ", " << dataPtr[chunkSize]
					<< ", " << dataPtr[chunkSize * 2] << std::endl;
			}

			if ((ite->key().find("fc") != std::string::npos) && (ite->key().find("weight") != std::string::npos)) {
				auto dataTensor = torch::randn(ite->value().sizes());
				dataTensor = dataTensor.div(sqrt(ite->value().numel()));
				auto rndPtr = dataTensor.data_ptr<float>();
				auto dataPtr = ite->value().data_ptr<float>();

				for (int j = 0; j < ite->value().numel(); j ++) {
					dataPtr[j] = rndPtr[j];
				}
				std::cout << "Initialized fc weight " << std::endl;
			}
		}
	}

	void loadModel(const std::string modelPath) {
		torch::serialize::InputArchive inChive;
		inChive.load_from(modelPath);
		this->load(inChive);
	}


	GRUMaskNet(int inSeqLen):
		gru0(torch::nn::GRUOptions(360, 2048).num_layers(2).batch_first(true)),
		fc(2048, 1),
		seqLen(inSeqLen)
	{
		register_module("gru0", gru0);
		register_module("fc", fc);

//		initParams();
	}

	Tensor inputPreprocess(Tensor input) {
		return input.div(4);
	}

	Tensor forward(vector<Tensor> inputs) {
		vector<int> seqLens(inputs.size(), 0);
		int total = 0;
		for (int i = 0; i < inputs.size(); i ++) {
			seqLens[i] = inputs[i].size(0);
			total += seqLens[i];

			inputs[i] = inputPreprocess(inputs[i]);
			inputs[i] = inputs[i].view({inputs[i].size(0), inputs[i].size(1) * inputs[i].size(2)}); //TODO: -1 is OK?
			inputs[i] = torch::constant_pad_nd(inputs[i], {0, 0, 0, (seqLen - seqLens[i])});
			seqLens[i] = std::min(seqLens[i], seqLen);
		}
		cout << "total input " << total << endl;

//		auto temp = torch::stack(inputs, 0);
		auto packedInput = torch::nn::utils::rnn::pack_padded_sequence(torch::stack(inputs, 0), torch::tensor(seqLens), true);
		auto gruOutput = gru0->forward_with_packed_input(packedInput);
		auto gruOutputData = std::get<0>(gruOutput);
		cout << "End of gru cell " << endl;

		auto unpacked = torch::nn::utils::rnn::pad_packed_sequence(gruOutputData, true);
		Tensor unpackedTensor = std::get<0>(unpacked);
		cout << "unpackedTensor: " << unpackedTensor.sizes() << endl;
		vector<Tensor> unpackDatas(seqLens.size());
		int outputTotal = 0;
		for (int i = 0; i < seqLens.size(); i ++) {
			Tensor unpackData = unpackedTensor[i].narrow(0, 0, seqLens[i]);
			unpackDatas[i] = unpackData;
			outputTotal += (int)unpackDatas[i].size(0);
		}
		cout << "Output total " << outputTotal << endl;

		Tensor fcInput = torch::cat(unpackDatas, 0);
		auto fcOutput = fc->forward(fcInput);
		cout << "End of fc cell " << endl;

		Tensor logmaxInput = fcOutput;
		Tensor logmaxOutput = torch::log_softmax(logmaxInput, -1);

		return logmaxOutput;
	}

//	Tensor forward(vector<Tensor> inputs) {
//		vector<int> seqLens(inputs.size(), 0);
//		int total = 0;
//		for (int i = 0; i < inputs.size(); i ++) {
//			seqLens[i] = inputs[i].size(0);
//			total += seqLens[i];
//
//			inputs[i] = inputPreprocess(inputs[i]);
//			inputs[i] = inputs[i].view({inputs[i].size(0), inputs[i].size(1) * inputs[i].size(2)}); //TODO: -1 is OK?
//			inputs[i] = torch::constant_pad_nd(inputs[i], {0, 0, 0, (seqLen - seqLens[i])});
//			seqLens[i] = std::min(seqLens[i], seqLen);
//		}
//		cout << "total input " << total << endl;
//
////		auto temp = torch::stack(inputs, 0);
//		auto packedInput = torch::nn::utils::rnn::pack_padded_sequence(torch::stack(inputs, 0), torch::tensor(seqLens), true);
//		auto gruOutput = gru0->forward_with_packed_input(packedInput);
//		auto gruOutputData = std::get<0>(gruOutput);
//		cout << "End of gru cell " << endl;
//
//		auto unpacked = torch::nn::utils::rnn::pad_packed_sequence(gruOutputData, true);
//		Tensor unpackedTensor = std::get<0>(unpacked);
//
//		Tensor lengths = std::get<1>(unpacked);
//		cout << "lengths ------------------------------> " << endl;
//		cout << lengths << endl;
//
//		cout << "unpackedTensor: " << unpackedTensor.sizes() << endl;
//		vector<Tensor> unpackDatas(seqLens.size());
//		int outputTotal = 0;
//		for (int i = 0; i < seqLens.size(); i ++) {
//			Tensor unpackData = unpackedTensor[i].narrow(0, 0, seqLens[i]);
//			unpackDatas[i] = unpackData;
//			outputTotal += (int)unpackDatas[i].size(0);
//		}
//		cout << "Output total " << outputTotal << endl;
//
//		Tensor fcInput = torch::cat(unpackDatas, 0);
//
////		Tensor fcInput = gruOutputData.data();
////		cout << "fcInput: " << fcInput.sizes() << endl;
//
//		auto fcOutput = fc->forward(fcInput);
//		cout << "End of fc cell " << endl;
//
//		Tensor logmaxInput = fcOutput;
//		Tensor logmaxOutput = torch::log_softmax(logmaxInput, 1);
//
//		return logmaxOutput;
//	}

	std::pair<Tensor, Tensor> forward (Tensor input, Tensor hState) {
		if (input.dim() < 3) {
			input = input.view({1, input.size(0), input.size(1)});
		}
		input = inputPreprocess(input);
//		if (hState.dim() < 3) {
//			hState = hState.view({1, hState.size(0), hState.size(1)});
//		}
		auto gruOutputData = gru0->forward(input, hState);
		auto gruOutput = std::get<0>(gruOutputData);
		auto newState = std::get<1>(gruOutputData);


		Tensor fcOutput = fc->forward(gruOutput);
		cout << "fcOutput: " << fcOutput.sizes() << endl;
		Tensor logmaxOutput = torch::log_softmax(fcOutput, -1);

		return std::make_pair(logmaxOutput, newState);
	}
};

}

const std::string wsPath = "/home/zf/";
std::ofstream dataFile(wsPath + "/workspaces/workspace_cpp/torchpractice/build/errorstats.txt");
const std::string modelPath = "/home/zf/workspaces/workspace_cpp/aws/GRU2L2048MaskNet_140000_0.002000_1593719779.pt";
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

	cout << "accu: " << accu << endl;
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
	Tensor output = net.forward(validInputs);

	auto loss = torch::nll_loss(output, labelTensor).item<float>();
	evalAccu(output, labelTensor);

}

static void evalStepModel(GRUMaskNet& net, LmdbSceneReader<LmdbDataDefs>& validReader, const int sampleNum) {
	validReader.reset();
	vector<Tensor> validInputs;
	vector<Tensor> validLabels;
	std::tie(validInputs, validLabels) = validReader.next(sampleNum);

	Tensor input = validInputs[0];
	Tensor label = validLabels[0];
	processActionInput(input, label);

	net.eval();
	const int seqLen = input.size(0);
	cout << "input sizes " << input.sizes() << endl;
	cout << "seqLen: " << seqLen << endl;

	vector<Tensor> outputs;
	outputs.reserve(seqLen);
	Tensor hState = torch::zeros({2, 1, 2048});
	for (int i = 0; i < seqLen; i ++) {
		Tensor netInput = input[i].view({1, 1, 360});
		auto output = net.forward(netInput, hState);
		hState = std::get<1>(output);
		outputs.push_back(std::get<0>(output).view({1, 42}));
	}

	cout << "output sizes " << outputs[0].sizes() << endl;
	Tensor outTensor = torch::cat(outputs, 0);
	cout << "outputTensor " << outTensor << endl;
	cout << "label " << label.sizes() << endl;

	evalAccu(outTensor, label);
}

static void testTrain(int sampleNum, int seqLen) {
	std::string validDbPath = wsPath + "/workspaces/res/dbs/lmdb5rowscenetestvalid";
	cout << "validDbPath: " << validDbPath << endl;
	LmdbSceneReader<LmdbDataDefs> validReader(validDbPath);

	GRUMaskNet net(seqLen);
	net.loadModel(modelPath);
	cout << "Model loaded " << endl;

	evalModel(net, validReader, sampleNum, seqLen);
//	evalStepModel(net, validReader, sampleNum);
}


static void testPack() {
	vector<Tensor> inputs;
	vector<int> lens;
	const int batchSize = 4;
	const int maxLen = batchSize;
	for (int i = 0; i < batchSize; i ++) {
		Tensor input = torch::ones({(maxLen - i), 8});
		input = input * (maxLen - i);
		inputs.push_back(input);
		lens.push_back((maxLen - i));
	}

	for (int i = 0; i < batchSize; i ++) {
		inputs[i] = torch::constant_pad_nd(inputs[i], {0, 0, 0, (maxLen - lens[i])});
	}

	Tensor inputTensor = torch::stack(inputs, 0);

	auto packedInput = torch::nn::utils::rnn::pack_padded_sequence(inputTensor, torch::tensor(lens), true);

	Tensor packedTensor = packedInput.data();

	auto unpacked = torch::nn::utils::rnn::pad_packed_sequence(packedInput, true);
	Tensor unpackedTensor = std::get<0>(unpacked);
	Tensor lengths = std::get<1>(unpacked);

	cout << "packed: ------------------------------> " << endl << endl;
	cout << packedTensor << endl;

	cout << "unpacked: ------------------------------> " << endl << endl;
	cout << unpackedTensor << endl;
}


int main(int argc, char** argv) {
	const int sampleNum = atoi(argv[1]);
	const int seqLen = 27;

	testTrain(sampleNum, seqLen);

//	testPack();
}


