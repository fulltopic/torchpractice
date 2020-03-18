/*
 * TextGenerator.cpp
 *
 *  Created on: Nov 21, 2019
 *      Author: zf
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>

#include <torch/torch.h>

#include "TextGenerator.h"

using namespace std;
using namespace torch;
using namespace torch::nn;

Net::Net(const int inputLen, const int hiddenLen, bool isSingleStep):
//		lstm(LSTM(LSTMOptions(inputLen, hiddenLen).batch_first(isSingleStep).layers(1).with_bias(true))),
//		lstm1(LSTM(LSTMOptions(hiddenLen, hiddenLen).batch_first(isSingleStep).with_bias(true))),
				lstm(LSTM(LSTMOptions(inputLen, hiddenLen).batch_first(true).layers(1).with_bias(true))),
				lstm1(LSTM(LSTMOptions(hiddenLen, hiddenLen).batch_first(true).with_bias(true))),
				fc(Linear(hiddenLen, inputLen)),
				singleStep(isSingleStep){
	register_module("lstm", lstm);
	register_module("lstm1", lstm1);
	register_module("fc", fc);
}

//Tensor Net::forward(Tensor input, const int seqLen, const int batchSize) {
//	Tensor state;
//
//	auto inputs = torch::chunk(input, input.size(0) / (seqLen * batchSize), 0);
//	vector<Tensor> outputs;
//	for (auto inputTensor: inputs) {
//		auto lstmOutput = lstm->forward(inputTensor.view({seqLen, batchSize, input.size(input.dim() - 1)}), state);
//		state = lstmOutput.state;
//		auto fcOutput = fc->forward(lstmOutput.output);
//		auto output = torch::log_softmax(fcOutput, fcOutput.dim() - 1);
//		outputs.push_back(output.view({seqLen * batchSize, input.size(input.dim() - 1)}));
//	}
//
//	return torch::cat(outputs, 0);
//}


Tensor Net::forward(Tensor input, const int seqLen, const int batchSize) {
	cout << "------------------------> Forward" << endl;
	Tensor state0;
	Tensor state1;

	const int hiddenSize = 200;

//	Tensor state0 = torch::zeros({2, 1, batchSize, hiddenSize});
//	Tensor state1 = torch::zeros({2, 1, batchSize, hiddenSize});

	auto lstmOutput0 = lstm->forward(input.view({seqLen, batchSize, input.size(input.dim() - 1)}), state0);
//	auto lstmOutput0 = lstm->forward(input.view({batchSize, seqLen, input.size(input.dim() - 1)}), state0);
	cout << "End of lstm first" << endl;

	state0 = lstmOutput0.state;
	auto lstmOutput1 = lstm1->forward(lstmOutput0.output, state1);
	cout << "End of lstm second " << endl;
	state1 = lstmOutput1.state;
	auto fcOutput = fc->forward(lstmOutput1.output);
	auto output = torch::log_softmax(fcOutput, fcOutput.dim() - 1);

	return output.view({seqLen * batchSize, input.size(input.dim() - 1)});
}

Tensor Net::forward(Tensor input, Tensor& state, Tensor& state1, const int seqLen, const int batchSize) {
	auto inputs = torch::chunk(input, input.size(0) / (seqLen * batchSize), 0);
//	cout << "inputs size " << inputs.size() << endl;
	vector<Tensor> outputs;
	for (auto inputTensor: inputs) {
		auto lstmOutput = lstm->forward(inputTensor.view({seqLen, batchSize, input.size(input.dim() - 1)}), state);
		state = lstmOutput.state;
		auto lstmOutput1 = lstm1->forward(lstmOutput.output, state1);
		state1 = lstmOutput1.state;
		auto fcOutput = fc->forward(lstmOutput1.output);
//		cout << "fcOutput " << fcOutput.sizes() << endl;
//		cout << fcOutput << endl;
		auto output = torch::log_softmax(fcOutput, fcOutput.dim() - 1);
//		cout << output << endl;
		outputs.push_back(output.view({seqLen * batchSize, input.size(input.dim() - 1)}));
	}

	return torch::cat(outputs, 0);
}

torch::Tensor Net::forwardSingleStep(torch::Tensor input, const int seqLen, const int batchSize) {
	if (!singleStep) {
		cout << "This net does not support single step forward " << endl;
		return torch::zeros({0});
	}

	vector<Tensor> outputs;
	Tensor state;
	const int seqIndex = 0;
	const int inputLen = input.size(input.dim() - 1);
	auto inputs = torch::chunk(input, seqLen, seqIndex);

	for (auto inputTensor: inputs) {
		auto lstmOutput = lstm->forward(inputTensor.view({batchSize, inputLen}), state);
		state = lstmOutput.state;
		auto fcOutput = fc->forward(lstmOutput.output);
		auto output = torch::log_softmax(fcOutput, fcOutput.dim());

		outputs.push_back(output);
	}

	return torch::cat(outputs, 0);
}


Shakespeare::Shakespeare(string filePath, int hiddenL):
		fileName(filePath),
		hiddenLen(hiddenL),
		net(nullptr)
{
	init();
	//TODO: Single step
	net = new Net(inputLen, hiddenL, false);
}

Shakespeare::~Shakespeare() {
	delete net;
	net = nullptr;
}

void Shakespeare::init() {
	ifstream infile(fileName);
	stringstream buffer;
	buffer << infile.rdbuf();
	text = buffer.str();

	set<char> vocab_set(text.begin(), text.end());
	vocab.assign(vocab_set.begin(), vocab_set.end());
	inputLen = vocab.size();

	auto index = 0;
	for (auto c : vocab) {
	    c2i[c] = index;
	    i2c[index++] = c;
	}

	cout << "End of init" << endl;
	cout << "Input has " << vocab.size()
			<< " characters. Total input size: " << text.size() << endl;

	for (auto p: i2c) {
		cout << p.first << " --> " << p.second << ", ";
	}
	cout << endl;
}

void Shakespeare::resetPos(const int batchSize) {
	startPos.resize(0);
	curPos.resize(0);

	sizePerBatch = text.length() / batchSize;
	for (int i = 0; i < batchSize; i ++) {
		startPos.push_back(i * sizePerBatch);
		curPos.push_back(0);
	}
}

pair<Tensor, Tensor> Shakespeare::generateInput(const int batchSize, const int seqLen, bool test) {
	Tensor inputTensor = torch::zeros({seqLen, batchSize, inputLen});
	Tensor targetTensor = torch::zeros({seqLen, batchSize, 1}).toType(ScalarType::Long);

	auto inputData = (float*)inputTensor.data_ptr();
	auto targetData = (long*)targetTensor.data_ptr();

//	cout << "Generate target ::::::::::::::::::::::::::::::::::::::" << endl;
//	cout << "The input seq =====================================> " << endl;
	for (int i = 0; i < batchSize; i ++) {
		for (int j = 0; j < seqLen; j ++) {
			int textPos = startPos[i] + curPos[i];
			inputData[j * batchSize * inputLen + i * inputLen + c2i[text.at(textPos)]] = 1;
			int targetPos = startPos[i] + ((curPos[i] + 1) % text.length());
			targetData[j * batchSize * 1 + i * 1] = c2i[text.at(targetPos)];
			curPos[i] = (curPos[i] + 1) % sizePerBatch;

//			if (i == 0) {
////				cout << text.at(targetPos);
//				int index = j * batchSize * inputLen;
//				for (int k = 0; k < inputLen; k ++) {
//					if (inputData[index + k] == 1) {
//						cout << i2c[k];
//					}
//				}
//			}
		}
	}
	cout << endl;

	return make_pair(inputTensor.view({seqLen * batchSize, inputLen}),
			targetTensor.view({seqLen * batchSize, 1}));
}

pair<Tensor, Tensor> Shakespeare::generateInput(int& pos, const int exampleNum) {
	Tensor inputTensor = torch::zeros({exampleNum, inputLen});
	Tensor targetTensor = torch::zeros({exampleNum, 1}).toType(ScalarType::Long);//.toType(ScalarType::Long);
	//TODO: Check input data, if it void type or float type
	auto inputData = (float*)inputTensor.data_ptr();
	auto targetData = (long*)targetTensor.data_ptr();

	for (int i = 0; i < exampleNum; i ++) {
		char c = text.at(pos % text.length());
		pos ++;

		inputData[inputLen * i + c2i[c]] = 1;
		targetData[i] = c2i[text.at((pos + 1) % text.length())];
	}

	return make_pair(inputTensor, targetTensor);
}

void Shakespeare::printText(Tensor output, Tensor target, const int seqLen, const int batchSize) {
	cout << "Sizes " << output.sizes() << ", " << target.sizes() << endl;

	auto outputData = (long*)output.view({output.numel()}).toType(ScalarType::Long).data_ptr();
	auto targetData = (long*)target.view({target.numel()}).toType(ScalarType::Long).data_ptr();

	cout << "Output " << endl;
	for (int i = 0; i < output.numel(); i += batchSize) {
		cout << outputData[i] << ", ";
//		cout << i2c[outputData[i]];
	}
	cout << endl;

	cout << "Target " << endl;
	for (int i = 0; i < target.numel(); i += batchSize) {
		cout << i2c[targetData[i]];
	}
	cout << endl;
}

//void  Shakespeare::train(const int epochNum, const int seqLen, const int batchSize, const int totalExampleNum) {
//	Tensor inputTensor;
//	Tensor targetTensor;
//	torch::optim::Adam optimizer (net->parameters(), torch::optim::AdamOptions(2.0));
//
//	const int iteNum = totalExampleNum / (seqLen * batchSize);
//
//	for (int i = 0; i < epochNum; i ++) {
//		int pos = 0;
//
//		//TODO: When to backward?
//		for (auto j = 0; j < iteNum; j ++) {
//			Tensor inputTensor;
//			Tensor targetTensor;
//			tie(inputTensor, targetTensor) = generateInput(pos, seqLen * batchSize);
//
//			optimizer.zero_grad();
//			auto output = net->forward(inputTensor, seqLen, batchSize);
//			auto loss = torch::nll_loss(output, targetTensor.view({targetTensor.size(0)}));
//			cout << "--------------------------> " << "Epoch" << i << " batch" << j << ": " << loss << endl;
//			printText(output.argmax(output.dim() - 1, false), targetTensor);
//
//			loss.backward();
//			optimizer.step();
//
//			generateText(100, 4);
//		}
//	}
//}

void  Shakespeare::train(const int epochNum, const int seqLen, const int batchSize, const int totalExampleNum) {
	Tensor inputTensor;
	Tensor targetTensor;
	torch::optim::Adam optimizer (net->parameters(), torch::optim::AdamOptions(0.005));
//	torch::optim::SGD optimizer (net->parameters(), torch::optim::SGDOptions(0.05).weight_decay(0.9999));

	const int iteNum = totalExampleNum / (seqLen * batchSize);


	for (int i = 0; i < epochNum; i ++) {
//		int pos = 0;
		resetPos(batchSize);

		//TODO: When to backward?
		for (auto j = 0; j < iteNum; j ++) {
			Tensor inputTensor;
			Tensor targetTensor;
			tie(inputTensor, targetTensor) = generateInput(batchSize, seqLen, false);
//			cout << "Input " << endl;
//			cout << inputTensor << endl;
//			cout << "Target " << endl;
//			cout << targetTensor << endl;

			optimizer.zero_grad();
			auto output = net->forward(inputTensor, seqLen, batchSize);
			auto loss = torch::nll_loss(output, targetTensor.view({targetTensor.size(0)}));
			cout << "--------------------------> " << "Epoch" << i << " batch" << j << ": " << loss << endl;
//			printText(output.argmax(output.dim() - 1, true), targetTensor, seqLen, batchSize);
//			auto printOutput = (float*)output.data_ptr();
//			for (int k = 0; k < seqLen; k ++) {
//				for (int t = 0; t < inputLen; t ++) {
//					int pos = k * batchSize * inputLen + 0 * inputLen + t;
//					cout << printOutput[pos] << ", ";
//				}
//				cout << endl;
//			}
//			auto printTensors = targetTensor.view({seqLen, batchSize, 1}).chunk(batchSize, 1);
//			cout << "All targets " << endl;
//			for (auto printTensor: printTensors) {
//				cout << "One target :::::::::::::::::::::::::::::::::::::::::::::::" << endl;
//				auto printData = (long*)printTensor.data_ptr();
//				for (int k = 0; k < seqLen; k ++) {
//					cout << i2c[printData[k]];
//				}
//				cout << endl;
//			}

			loss.backward();
			optimizer.step();

			if (j % 10 == 0) {
				generateText(100, 4);
			}
		}
	}
}

void Shakespeare::generateText(const int textLen, const int candidateNum) {
	stringstream textStream;
	int64_t chIdx = rand() % i2c.size();
	textStream << i2c[chIdx];
	Tensor state;
	Tensor state1;

	for (int i = 0; i < textLen; i ++) {
		Tensor input = torch::zeros({1, 1, inputLen});
		auto ptr = input.data_ptr<float>();
		ptr[chIdx] = 1;

		auto output = net->forward(input, state, state1, 1, 1);
		auto sortOutput = output.argsort(output.dim() - 1, true);
		chIdx = ((long*)sortOutput.data_ptr())[rand() % candidateNum];
		textStream << i2c[chIdx];
	}

	cout << "Generated text " << endl;
	cout << textStream.str();
	cout << endl << endl;
}

void testEmbed() {
	Tensor tensor = torch::zeros({2, 2, 1}).toType(ScalarType::Long);

	auto data = (long*)tensor.data_ptr();
	int k = 0;
	int inputLen = 6;
	for (int i = 0; i < tensor.size(0); i ++) {
		for (int j = 0; j < tensor.size(1); j ++) {
			data[i * tensor.size(1) + j] = k % inputLen;
		}
	}

	torch::nn::Embedding tester(inputLen, inputLen);

	tester->weight = torch::zeros({inputLen});
	auto wData = tester->weight.data();

	auto output = tester->forward(tensor);

	cout << output << endl;
}

int main() {
	string fileName = "./data/shakespeare.txt";
//	string fileName = "./data/world.txt";
	Shakespeare tester(fileName, 200);
	int seqLen = 32;
	int batchSize = 32;
	int iteNum = 64;
	int epochNum = 80;
	tester.train(epochNum, seqLen, batchSize, seqLen * batchSize * iteNum);

//	testEmbed();
}
