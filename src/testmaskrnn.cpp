/*
 * testmaskrnn.cpp
 *
 *  Created on: May 22, 2020
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

using std::vector;
using torch::Tensor;
using std::cout;
using std::endl;

static bool tensorSeqComp(const Tensor& t0, const Tensor& t1) {
	return t0.size(0) > t1.size(0);
}

static void testPad() {
	const std::string dbPath = "/home/zf/workspaces/res/dbs/lmdbscenetest";
	LmdbSceneReader<LmdbDataDefs> reader(dbPath);

	std::vector<Tensor> inputs;
	std::vector<Tensor> labels;
	std::tie(inputs, labels) = reader.next(2);

	cout << inputs.size() << endl;
	cout << inputs[0].sizes() << endl;
	cout << inputs[1].sizes() << endl;

	std::sort(inputs.begin(), inputs.end(), tensorSeqComp);
	cout << "sorted " << endl;
	for (auto input: inputs) {
		cout << input.sizes() << endl;
	}
//	const int seqLen = inputs[0].size(0);
	const int seqLen = 14;
	vector<int> lens(inputs.size(), 0);
//	for (int i = 0; i < lens.size(); i ++) {
//		lens[i] = inputs[i].size(0);
//	}
	for (int i = 0; i < inputs.size(); i ++) {
		lens[i] = std::min((int)inputs[i].size(0), seqLen);
		int inputSeqLen = inputs[i].size(0);
		inputs[i] = inputs[i].view({inputs[i].size(0), inputs[i].size(1) * inputs[i].size(2)});
		inputs[i] = torch::constant_pad_nd(inputs[i], {0, 0, 0, (seqLen - inputSeqLen)});
		cout << "Padded " << inputs[i].sizes() << endl;
//		cout << inputs[i] << endl;
	}

	Tensor packInput = torch::stack(inputs, 0);
//	torch::nn::utils::rnn::PackedSequence packSeq(packInput, torch::tensor(lens));
	auto packed = torch::nn::utils::rnn::pack_padded_sequence(packInput, torch::tensor(lens), true, seqLen);
	cout << "Get packed input " << endl;
//	cout << packed.batch_sizes() << endl;
//	cout << packed.data() << endl;
//	const Tensor data = packed.data();
//	const Tensor batchSizes = packed.batch_sizes();
//	cout << "Packed " << endl;
//	cout << data.sizes() << endl;
//	cout << batchSizes.sizes() << endl;


	torch::nn::GRU gru(torch::nn::GRUOptions(360, 1024).batch_first(true));
	auto gruOutput = gru->forward_with_packed_input(packed);
	auto output = std::get<0>(gruOutput);
	auto state = std::get<1>(gruOutput);
	cout << "Output " << output.data().sizes() << endl;
	cout << "State " << state.sizes() << endl;

	auto paddedInput = torch::nn::utils::rnn::pad_packed_sequence(output, true);
	auto batchNormInput = std::get<0>(paddedInput);
//	torch::nn::BatchNorm1d batchNorm(seqLen);
//	auto batchNormOutput = batchNorm->forward(batchNormInput);
//	cout << "batchNormOutput " << batchNormOutput.sizes() << endl;

	torch::nn::Linear fc(1024, 42);
	auto fcOutput = fc->forward(batchNormInput);
//	auto fcOutput = fc->forward(batchNormOutput);
	cout << fcOutput.sizes() << endl;

	auto packedOutput = torch::nn::utils::rnn::pack_padded_sequence(fcOutput, torch::tensor(lens), true);
	cout << packedOutput.data().sizes() << endl;
	Tensor logmaxInput = packedOutput.data();
	Tensor logmaxOutput = torch::log_softmax(logmaxInput, 1);
	cout << "logmaxOutput: " << logmaxOutput.sizes() << endl;

	for (int i = 0; i < labels.size(); i ++) {
		if (labels[i].size(0) > lens[i]) {
			labels[i] = labels[i].narrow(0, 0, lens[i]);
		}
	}
	Tensor labelTensor = torch::cat(labels, 0);
	auto loss = torch::nll_loss(logmaxOutput, labelTensor);
	cout << loss.item<float>() << endl;
}

int main() {
	testPad();
}
