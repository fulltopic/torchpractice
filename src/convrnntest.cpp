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

#include "nets/lstmnet.h"
#include "nets/fcnet.h"
#include "nets/FixedFcNet.h"
#include "nets/purefcnet.h"

//#include "pytools/plotserver.h"
#include "pytools/batchplotserver.h"

using Tensor = torch::Tensor;
using TensorList = torch::TensorList;
using string = std::string;

//bool compare(const Tensor& t0, const Tensor& t1) {
//	return t0.size(0) > t1.size(0);
//};
//
//std::ofstream dataFile;
//
//struct Net: torch::nn::Module {
//	torch::nn::Conv2d conv0;
//	torch::nn::Conv2d conv1;
//	torch::nn::LSTM lstm0;
//	torch::nn::Linear fc;
//	long totalLen;
//	long totalSample;
//
//	Net(): conv0(torch::nn::Conv2dOptions(Conv0InChan, Conv0OutChan, {Conv0KernelH, Conv0KernelW}).stride({Conv0StrideH, Conv0StrideW})),
//			conv1(torch::nn::Conv2dOptions(Conv1InChan, Conv1OutChan, {Conv1KernelH, Conv1KernelW})),
//			lstm0(torch::nn::LSTM(torch::nn::LSTMOptions(GetLstm0Input(), Lstm0Hidden).batch_first(true))),
//			fc(torch::nn::Linear(FcInput, FcOutput)),
//			totalLen(0),
//			totalSample(0)
//	{
//		register_module("conv0", conv0);
//		register_module("conv1", conv1);
//		register_module("lstm0", lstm0);
//		register_module("fc", fc);
//	}
//
//	//TODO: Currently,  batch_size = 1
//	//TODO: seqLen
//	//TODO: dropout
//	Tensor forward (Tensor input) {
//		Tensor lstmState;
//		input.set_requires_grad(true);
//		const int seqLen = input.size(0);
//		const int batchSize = input.size(0) / seqLen;
//
//		input = input.view({input.size(0), InputC, input.size(1), input.size(2)});
//		std::cout << "input " << input.sizes() << std::endl;
//		auto conv0Output = conv0->forward(input);
//		std::cout << "conv0 output " << conv0Output.sizes() << std::endl;
//		conv0Output = torch::relu(torch::max_pool2d(conv0Output, Conv0PoolSize));
//		std::cout << "conv0 pool " << conv0Output.sizes() << std::endl;
//
//		auto conv1Output = conv1->forward(conv0Output);
//		std::cout << "conv1 output " << conv1Output.sizes() << std::endl;
//		conv1Output = torch::relu(torch::max_pool2d(conv1Output, Conv1PoolSize));
//		std::cout << "conv1 pool " << conv1Output.sizes() << std::endl;
//
//		auto lstmInput = conv1Output.view({batchSize, seqLen, conv1Output.size(1) * conv1Output.size(2) * conv1Output.size(3)});
//		auto lstmOutput = lstm0->forward(lstmInput, lstmState);
//		lstmState = lstmOutput.state;
//
//		auto fcOutput = fc->forward(lstmOutput.output);
//		fcOutput = torch::relu(fcOutput);
//		std::cout << "fcoutput " << fcOutput.sizes() << std::endl;
//
//		//TODO: Not a good output
//		return torch::log_softmax(fcOutput, 1);
//	}
//
//
////	const int batchSize = 2;
//	Tensor forward(std::vector<Tensor> inputs) {
////		inputs.push_back(torch::ones({10, 5, 72}));
//
//		const int batchSize = inputs.size();
//		std::cout << "batchSize " << batchSize << std::endl;
//
//		int maxSeqLen = 0;
//		for (auto it = inputs.begin(); it != inputs.end(); it ++) {
//			if (maxSeqLen < it->size(0)) {
//				maxSeqLen = it->size(0);
//			}
//		}
//		std::cout << "maxSeqLen: " << maxSeqLen << std::endl;
//
//		std::vector<Tensor> convOutputs;
//		for (int i = 0; i < inputs.size(); i ++) {
//			Tensor input = inputs[i];
//			input = input.view({input.size(0), InputC, input.size(1), input.size(2)});
//			auto conv0Output = conv0->forward(input);
//			conv0Output = torch::relu(torch::max_pool2d(conv0Output, {Conv0PoolH, Conv0PoolW}, {}, {1, 0}));
//
//			auto conv1Output = conv1->forward(conv0Output);
//			conv1Output = torch::relu(torch::max_pool2d(conv1Output, Conv1PoolSize));
//
//			//The shape is {seqLen, channel * 2d}
//			convOutputs.push_back(
//					conv1Output.view({conv1Output.size(0), conv1Output.size(1) * conv1Output.size(2) * conv1Output.size(3)}));
//		}
//		std::sort(convOutputs.begin(), convOutputs.end(), compare);
//		std::cout << "After sort " << std::endl;
//
//		std::vector<int64_t> lengthVec(convOutputs.size(), 0);
//		for (int i = 0; i < convOutputs.size(); i ++) {
//			lengthVec[i] = convOutputs[i].size(0);
//			std::cout << "length " << lengthVec[i] << std::endl;
//
//			convOutputs[i] = torch::constant_pad_nd(convOutputs[i], {0, 0, 0, (maxSeqLen - lengthVec[i])}, 0);
//		}
//
//		Tensor lstmInput = at::stack(convOutputs, 0);
//		std::cout << lstmInput.sizes() << std::endl;
//		std::cout << "lengths " << std::endl;
//		std::cout << torch::tensor(lengthVec) << std::endl;
//
//		Tensor lstmInputData;
//		Tensor lstmInputBatch;
//		std::tie(lstmInputData, lstmInputBatch) = at::_pack_padded_sequence(
//				at::stack(convOutputs, 0), torch::tensor(lengthVec), true);
//		std::cout << "lstmInputData: " << lstmInputData.sizes() << std::endl;
//		std::cout << "lstmInputBatch: " << lstmInputBatch.sizes() << std::endl;
//		std::cout << lstmInputBatch << std::endl;
//
//		Tensor lstmState = torch::zeros({2, Lstm0Layer * Lstm0Dir, (long)(lengthVec.size()), Lstm0Hidden});
//		std::cout << "State sizes: " << lstmState.sizes() << std::endl;
//		lstm0->enablePacked(lstmInputBatch);
//		auto lstmRnnOutput = lstm0->forward(lstmInputData, lstmState);
//		lstmState = lstmRnnOutput.state;
//		auto lstmOutput = lstmRnnOutput.output;
//		std::cout << "lstmOutput " << lstmOutput.sizes() << std::endl;
//
////		Tensor test0;
////		Tensor test1;
////		std::tie(test0, test1) = at::_pad_packed_sequence(lstmOutput, lstmInputBatch, true, 0, 0);
////		std::cout << "test0 " << test0.sizes() << std::endl;
////		std::cout << "test1 " << test1.sizes() << std::endl;
////		std::cout << test1 << std::endl;
//
//		Tensor fcOutput = fc->forward(lstmOutput);
//		std::cout << "fcOutput " << fcOutput.sizes() << std::endl;
//
//		Tensor output = torch::log_softmax(fcOutput, OutputMax);
//
//		return output;
//	}
//
//	Tensor forward(std::vector<Tensor> inputs, bool testLen) {
//		for (int i = 0; i < inputs.size(); i ++) {
//			totalLen += inputs[i].size(0);
//			totalSample ++;
//		}
//
//		return inputs[0];
//	}
//
//	Tensor forward(std::vector<Tensor> inputs, const int SeqLen) {
//		std::vector<Tensor> convOutputs;
//
//		for (int i = 0; i < inputs.size(); i ++) {
//			Tensor input = inputs[i];
//
////			if (input.size(0) < SeqLen) {
////				continue;
////			} else if (input.size(0) > SeqLen) {
////				input = input.narrow(0, input.size(0) - SeqLen, SeqLen);
////			}
//
//			std::cout << "Narrowed input " << input.sizes() << std::endl;
//
//			input = input.view({input.size(0), InputC, input.size(1), input.size(2)});
//			auto conv0Output = conv0->forward(input);
//			conv0Output = torch::relu(torch::max_pool2d(conv0Output, {Conv0PoolH, Conv0PoolW}, {}, {Conv0PoolPadH, Conv0PoolPadW}));
//
//			auto conv1Output = conv1->forward(conv0Output);
//			conv1Output = torch::relu(torch::max_pool2d(conv1Output, Conv1PoolSize));
//
//			//The shape is {seqLen, channel * 2d}
//			convOutputs.push_back(
//					conv1Output.view({conv1Output.size(0), conv1Output.size(1) * conv1Output.size(2) * conv1Output.size(3)}));
//		}
////		std::sort(convOutputs.begin(), convOutputs.end(), compare);
////		std::cout << "After sort " << std::endl;
////
////		std::vector<int64_t> lengthVec(convOutputs.size(), 0);
////		for (int i = 0; i < convOutputs.size(); i ++) {
////			lengthVec[i] = convOutputs[i].size(0);
////			std::cout << "length " << lengthVec[i] << std::endl;
////
////			convOutputs[i] = torch::constant_pad_nd(convOutputs[i], {0, 0, 0, (maxSeqLen - lengthVec[i])}, 0);
////		}
//
//		Tensor lstmInput = at::stack(convOutputs, 0);
//		std::cout << "lstminput " << lstmInput.sizes() << std::endl;
////		std::cout << "lengths " << std::endl;
////		std::cout << torch::tensor(lengthVec) << std::endl;
//
////		Tensor lstmInputData;
////		Tensor lstmInputBatch;
////		std::tie(lstmInputData, lstmInputBatch) = at::_pack_padded_sequence(
////				at::stack(convOutputs, 0), torch::tensor(lengthVec), true);
////		std::cout << "lstmInputData: " << lstmInputData.sizes() << std::endl;
////		std::cout << "lstmInputBatch: " << lstmInputBatch.sizes() << std::endl;
////		std::cout << lstmInputBatch << std::endl;
//
//		Tensor lstmState = torch::zeros({2, Lstm0Layer * Lstm0Dir, (long)(convOutputs.size()), Lstm0Hidden});
//		std::cout << "State sizes: " << lstmState.sizes() << std::endl;
////		lstm0->enablePacked(lstmInputBatch);
//		auto lstmRnnOutput = lstm0->forward(lstmInput, lstmState);
//		lstmState = lstmRnnOutput.state;
//		auto lstmOutput = lstmRnnOutput.output;
//		std::cout << "lstmOutput " << lstmOutput.sizes() << std::endl;
//
////		Tensor test0;
////		Tensor test1;
////		std::tie(test0, test1) = at::_pad_packed_sequence(lstmOutput, lstmInputBatch, true, 0, 0);
////		std::cout << "test0 " << test0.sizes() << std::endl;
////		std::cout << "test1 " << test1.sizes() << std::endl;
////		std::cout << test1 << std::endl;
//
//		Tensor fcOutput = fc->forward(lstmOutput);
//		std::cout << "fcOutput " << fcOutput.sizes() << std::endl;
//
//		Tensor output = torch::log_softmax(fcOutput, OutputMax);
//
//		return output.view({output.size(0) * output.size(1), output.size(2)});
//
//	}
//
//	Tensor forward(std::vector<Tensor> inputs, const int seqLen, bool toRecord) {
////		std::cout << "Start forward " << std::endl;
//		std::vector<Tensor> convOutputs;
//		std::vector<Tensor> inputView;
//
//		for (int i = 0; i < inputs.size(); i ++) {
//			Tensor input = inputs[i];
////			std::cout << "Input " << input.sizes() << std::endl;
//			inputView.push_back(input.view({input.size(0), InputC, input.size(1), input.size(2)}));
//		}
////		std::cout << "End of view" << std::endl;
//
//		Tensor conv0Input = at::cat(inputView, 0);
////		std::cout << "conv0Input " << conv0Input.sizes() << std::endl;
//
//		auto conv0Output = conv0->forward(conv0Input);
//		conv0Output = torch::relu(torch::max_pool2d(conv0Output, {Conv0PoolH, Conv0PoolW}, {}, {Conv0PoolPadH, Conv0PoolPadW}));
//
//		auto conv1Output = conv1->forward(conv0Output);
//		if (toRecord) {
//			dataFile << "Conv0ouput -------------------------------------------> " << std::endl;
//			dataFile << conv1Output << std::endl;
//		}
//		conv1Output = torch::relu(torch::max_pool2d(conv1Output, Conv1PoolSize));
//		if (toRecord) {
//			dataFile << "After pool -------------------------------------------> " << std::endl;
//			dataFile << conv1Output << std::endl;
//			dataFile << "============================================> END ==================================> " << std::endl;
//		}
//		//		std::cout << "Conv output: " << conv1Output.sizes() << std::endl;
//
////		Tensor lstmInput = at::stack(convOutputs, 0);
//		Tensor lstmInput = conv1Output.view(
//				{conv1Output.size(0) / seqLen, seqLen, conv1Output.size(1) * conv1Output.size(2) * conv1Output.size(3)});
////		std::cout << "lstminput: " << lstmInput.sizes() << std::endl;
//
////		Tensor lstmState = torch::zeros({2, Lstm0Layer * Lstm0Dir, (long)(convOutputs.size()), Lstm0Hidden});
////		std::cout << "State sizes: " << lstmState.sizes() << std::endl;
//
//		Tensor lstmState;
//		auto lstmRnnOutput = lstm0->forward(lstmInput, lstmState);
//		lstmState = lstmRnnOutput.state;
//		auto lstmOutput = lstmRnnOutput.output;
////		std::cout << "lstmOutput " << lstmOutput.sizes() << std::endl;
//
//		Tensor fcOutput = fc->forward(lstmOutput);
////		std::cout << "fcOutput " << fcOutput.sizes() << std::endl;
//
//		Tensor output = torch::log_softmax(fcOutput, OutputMax);
//
//		return output.view({output.size(0) * output.size(1), output.size(2)});
//
////		return inputs[0];
//	}
//};

void train(LmdbSceneReader<LmdbDataDefs>& reader, LstmNet& net) {
	torch::optim::Adam optimizer (net.parameters(), torch::optim::AdamOptions(0.005));

	Tensor input;
	Tensor label;

	optimizer.zero_grad();

	std::tie(input, label) = reader.next();
	auto output = net.forward(input);
	std::cout << "output " << output.sizes() << std::endl;
//	std::cout << output << std::endl;
	output = output.view({output.size(0) * output.size(1), output.size(2)});
	std::cout << "Viewed output " << output.sizes() << std::endl;

	std::cout << "label " << label.sizes() << std::endl;
	auto loss = torch::nll_loss(output, label);
	loss.backward();
	optimizer.step();
}

//void writeConvWeight(Tensor w) {
//	for (int i = 0; i < w.size(0); i ++) {
//		for (int j = 0; j < w.size(3); j ++) {
//			dataFile << *(w[i][0][0][j].data<float>()) << ", ";
//		}
//		dataFile << std::endl;
//	}
//	dataFile << "-------------------------------------------------> End of weight ---------------------------> " << std::endl;
//}

void train(LmdbSceneReader<LmdbDataDefs>& reader, LstmNet& net, const int batchSize, torch::optim::Optimizer& optimizer) {
//	torch::optim::Adam optimizer (net.parameters(), torch::optim::AdamOptions(0.01));


	optimizer.zero_grad();

	std::cout << "conv0 weight shape " << 	net.conv0->weight.sizes() << std::endl;

	while (reader.hasNext()) {
		std::vector<Tensor> inputs;
		std::vector<Tensor> labels;

		std::tie(inputs, labels) = reader.next(batchSize);
		Tensor output = net.forward(inputs);
		std::cout << "train output " << output.sizes() << std::endl;

		std::cout << "Labels " << labels.size() << std::endl;
		Tensor labelTensor = at::cat(labels, 0);
		std::cout << "Label tensor: " << labelTensor.sizes() << std::endl;

		auto loss = torch::nll_loss(output, labelTensor);
		std::cout << "====================================================> loss " << loss << std::endl;

		loss.backward();
		optimizer.step();

//		writeConvWeight(net.conv0->weight);
	}
//	std::cout << "Got " << inputs.size() << std::endl;
//	std::cout << inputs[0].device() << std::endl;

//	dataFile.close();
}


float evaluation(Tensor outputs, Tensor labels, const int seqLen) {
	const int batchSize = outputs.size(0) / seqLen;
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
	const long* labelData = labels.data<long>();
	const long* indexData = indices.data<long>();

//	int matched = 0;
//	int total = labels.size(0);
//
//	for (int i = 0; i < total; i ++) {
//		if (labelData[i] == indexData[i]){
//			matched ++;
//		}
//	}

	float accu = (float)matched / total;
//	std::cout << "Accuracy: " << matched << "/" << total << " = " << (float)matched / total << std::endl;
//
//	for (int i = 0; i < seqLen; i ++) {
//		std::cout << indexData[i] << "-->" << labelData[i] << ", ";
//	}
//	std::cout << std::endl;

	return accu;
}

void printWeights(Tensor ws, const std::string dirPath, const int h, const int w) {
	auto wcs = torch::chunk(ws, ws.size(0), 0);
	for (int i = 0; i < wcs.size(); i ++) {
		//TODO: The size of kernel
		matplotlibcpp::imshow(wcs[i].data<float>(), h, w, 1, std::map<std::string, std::string>());
		std::string imgName = dirPath + "/test" + std::to_string(i) + ".png";
		matplotlibcpp::save(imgName);
	}
}

template<typename NetType>
typename std::enable_if<!std::is_same<NetType, PureFcNet>::value, bool>::type
 printConvWeight(const int count, NetType& net) {
	const std::string dirPath = "./images/" + std::to_string(count) + "/";
	std::cout << "To create " << dirPath << std::endl;
	printWeights(net.conv0->weight, dirPath, net.conv0->weight.size(2), net.conv0->weight.size(3));

	std::cout << "is same " << std::is_same<NetType, PureFcNet>::value << std::endl;

	return true;
}

template<typename NetType>
typename std::enable_if<std::is_same<NetType, PureFcNet>::value, bool>::type
printConvWeight(const int count, NetType& net) {
	return true;
}

template<typename NetType>
void trainSameLen(LmdbSceneReader<LmdbDataDefs>& reader, NetType& net, const int batchSize, torch::optim::Optimizer& optimizer) {
//	torch::optim::Adam optimizer (net.parameters(), torch::optim::AdamOptions(0.01));
	const int len = 13;
	optimizer.zero_grad();
	int64_t totalNum = 0;
	double totalLoss = 0;
	int count = 0;

	std::cout << "conv0 weight shape " << 	net.conv0->weight.sizes() << std::endl;

	while (reader.hasNext()) {
		std::vector<Tensor> inputs;
		std::vector<Tensor> labels;
		std::vector<Tensor> validLabels;
		std::vector<Tensor> validInputs;

		std::tie(inputs, labels) = reader.next(batchSize, len);

		for (int i = 0; i < labels.size(); i ++) {
			inputs[i] = torch::div(inputs[i], 4);

			if (labels[i].size(0) < len) {
				continue;
			} else if (labels[i].size(0) > len) {
				validLabels.push_back(labels[i].narrow(0, labels[i].size(0) - len, len));
				validInputs.push_back(inputs[i].narrow(0, inputs[i].size(0) - len, len));
			} else {
				validLabels.push_back(labels[i]);
				validInputs.push_back(inputs[i]);
			}

//			std::cout << "One inputs " << validInputs[i] << std::endl;
		}
		if (validLabels.size() == 0) {
			continue;
		}
		totalNum += validLabels.size();

		std::vector<int> index(validInputs.size(), 0);
		for (int i = 0; i < index.size(); i ++) {
			index[i] = i;
		}
		std::random_shuffle(index.begin(), index.end());
		inputs.clear();
		labels.clear();
		for (int i = 0; i < index.size(); i ++) {
			inputs.push_back(validInputs[index[i]]);
			labels.push_back(validLabels[index[i]]);
		}

//		Tensor output = net.forward(inputs, len);
		Tensor output = net.forward(inputs, len, false);
//		std::cout << "train output " << output.sizes() << std::endl;

//		std::cout << "Labels " << validLabels.size() << std::endl;
//		Tensor labelTensor = at::cat(validLabels, 0);
//		std::cout << "Label tensor: " << labelTensor.sizes() << std::endl;
//		std::cout << "Labels " << labels.size() << std::endl;
		Tensor labelTensor = at::cat(labels, 0);
//		std::cout << "Label tensor: " << labelTensor.sizes() << std::endl;
		auto loss = torch::nll_loss(output, labelTensor);
		totalLoss += (loss.item<float>() * validInputs.size());
		auto accu = evaluation(output, labelTensor, len);
		std::cout << "====================================================> loss " << loss.item<float>()
				<< ", " << (totalLoss / totalNum)
				<< ", " << accu << std::endl;

		if ((count % 128) == 0) {
			printConvWeight(count, net);
		}

		count ++;
		loss.backward();
		optimizer.step();

//		writeConvWeight(net.conv0->weight);
	}
//	std::cout << "Got " << inputs.size() << std::endl;
//	std::cout << inputs[0].device() << std::endl;

//	dataFile.close();
}

template<typename DataType>
void shuffleInput(std::vector<DataType>& inputs, std::vector<DataType>& labels) {
	std::vector<DataType> validLabels(labels.begin(), labels.end());
	std::vector<DataType> validInputs(inputs.begin(), inputs.end());

	std::vector<int> index(validInputs.size(), 0);
	for (int i = 0; i < index.size(); i ++) {
		index[i] = i;
	}
	std::random_shuffle(index.begin(), index.end());
	inputs.clear();
	labels.clear();
	for (int i = 0; i < index.size(); i ++) {
		inputs.push_back(torch::div(validInputs[index[i]], 4));
//		inputs.push_back(validInputs[index[i]]);
		labels.push_back(validLabels[index[i]]);
	}
//	std::cout << inputs[0] << std::endl;
}

template<typename DbDefs, typename NetType>
std::pair<float, float> validOverfit(LmdbSceneReader<DbDefs>& reader, NetType& net, const int seqLen, const int dbStartPos) {
//	const int dbStartPos = 1000;
	const int batchSize = 8;
	reader.reset();
	int pos = 0;
	while (reader.hasNext() && (pos < dbStartPos)) {
		reader.next();
		pos ++;
	}

	std::vector<Tensor> inputs;
	std::vector<Tensor> labels;
	std::tie(inputs, labels) = reader.next(batchSize, seqLen);
	shuffleInput(inputs, labels);
	Tensor labelTensor = torch::cat(labels, 0);
	Tensor output = net.forward(inputs, seqLen, false, false);

	auto loss = torch::nll_loss(output, labelTensor).item<float>();
	auto accu = evaluation(output, labelTensor, seqLen);

	return std::make_pair(loss, accu);
}



std::vector<float> plotLoss(1000, 0);
std::vector<float> plotAccu(1000, 0);
int lossIndex = 0;
int accuIndex = 0;
const int cap = plotLoss.size();
const int threshold = cap / 2;
void adjustPlotData(std::vector<float>& data, int&index, float newData) {
	if (index >= cap) {
		for (int i = 0; i < threshold; i ++) {
			data[i] = data[i * 2];
		}
		index = threshold;

		for (int i = threshold; i < cap; i ++) {
			data[i] = 0;
		}
	}
	data[index] = newData;
	index ++;
}
void plotStats(float loss, float accu) {
	adjustPlotData(plotLoss, lossIndex, loss);
	adjustPlotData(plotAccu, accuIndex, accu);

	matplotlibcpp::clf();
	matplotlibcpp::subplot(1, 2, 1);
	matplotlibcpp::plot(plotLoss);
	matplotlibcpp::subplot(1, 2, 2);
	matplotlibcpp::plot(plotAccu);
	matplotlibcpp::pause(0.01);
}


template<typename NetType, typename DbDefs>
void trainOverfit(LmdbSceneReader<DbDefs>& reader, NetType& net, torch::optim::Optimizer& optimizer) {
//	torch::optim::Adam optimizer (net.parameters(), torch::optim::AdamOptions(0.01));
	const int len = 8;
	optimizer.zero_grad();
	int64_t totalNum = 0;
	double totalLoss = 0;
	const int epoch = 512;
	const int batchSize = 8;
	const int sampleNum = 80000;
	const int validStep = 128;
	const int earlyStop = 16;
	int stopStep = 0;
	float lastLoss = FLT_MAX;
	int count = 0;
//	std::cout << "conv0 weight shape " << 	net.conv0->weight.sizes() << std::endl;
	std::vector<float> iteVec;
	std::vector<float> lossVec;
	std::map<std::string, std::string> keys;

	auto& plotServer = BatchPlotServer::GetInstance();
	std::cout << "Get server instance" << std::endl;
//	PlotServer::Start();
	std::thread t(BatchPlotServer::Run);
	std::cout << "Start thread " << std::endl;


	float iteNum = 1;



	while (count <= epoch) {
		totalNum = 0;
		totalLoss = 0;
		bool printed = false;
		reader.reset();
		while (totalNum < sampleNum) {
			std::vector<Tensor> inputs;
			std::vector<Tensor> labels;

			std::tie(inputs, labels) = reader.next(batchSize, len);
			shuffleInput(inputs, labels);
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
			auto loss = torch::nll_loss(output, labelTensor);
			totalLoss += (loss.item<float>() * inputs.size());
//			std::cout << "End of loss " << std::endl;
			auto accu = evaluation(output, labelTensor, len);
			std::cout << "====================================================> loss " << loss.item<float>()
				<< ", " << (totalLoss / totalNum)
				<< ", " << accu << std::endl;

			plotServer.newEvent(std::make_pair(TrainLossIndex, loss.item<float>()));
//			std::cout << "Push event " << std::endl;
			plotServer.newEvent(std::make_pair(TrainAccuIndex, accu));
//			std::cout << "Push another event" << std::endl;


//			plotStats(loss.item<float>(), accu);
//			iteVec.push_back(iteNum);
//			lossVec.push_back(loss.item<float>());
//			iteNum ++;
//			matplotlibcpp::plot(iteVec, lossVec, keys);
//			matplotlibcpp::pause(0.01);


			loss.backward();
			optimizer.step();
		}
//		if ((count % validStep) == 0) {
//			printConvWeight(count, net);
//
//			auto validLoss = validOverfit(reader, net, len, (sampleNum + 10));
//			std::cout << "---------------------------------> Validation loss: " << std::get<0>(validLoss)
//					<< ", " << std::get<1>(validLoss) << std::endl;
//		}

		auto validLoss = validOverfit(reader, net, len, (sampleNum + 100));
		std::cout << "---------------------------------> Validation loss: " << std::get<0>(validLoss)
				<< ", " << std::get<1>(validLoss) << std::endl;

		plotServer.newEvent(std::make_pair(ValidLossIndex, std::get<0>(validLoss)));
		plotServer.newEvent(std::make_pair(ValidAccuIndex, std::get<1>(validLoss)));

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

	plotServer.notifyRead();
	plotServer.stop();
	t.join();
//	std::cout << "conv0 weights " << std::endl;
//	std::cout << net.conv0->weight << std::endl;

//	printConvWeight(epoch, net);
}


template<typename NetType>
std::pair<float, float> validOverfitThread(DataType& datas, NetType& net, const int seqLen) {
//	const int dbStartPos = 1000;

	std::vector<Tensor> inputs;
	std::vector<Tensor> labels;
	std::tie(inputs, labels) = datas;
	shuffleInput(inputs, labels);
	Tensor labelTensor = torch::cat(labels, 0);
	Tensor output = net.forward(inputs, seqLen, false, false);

	auto loss = torch::nll_loss(output, labelTensor).item<float>();
	auto accu = evaluation(output, labelTensor, seqLen);

	return std::make_pair(loss, accu);
}

template<typename DbDefs>
void runReader(ReaderWrapper<DbDefs>& reader) {
	reader.start();
}

template<typename NetType, typename DbDefs>
void trainOverfitThread(const std::string dbPath, NetType& net, torch::optim::Optimizer& optimizer) {
//	torch::optim::Adam optimizer (net.parameters(), torch::optim::AdamOptions(0.01));
	const int len = 8;
	optimizer.zero_grad();
	int64_t totalNum = 0;
	double totalLoss = 0;
	const int epoch = 512;
	const int batchSize = 8;
	const int sampleNum = 32;
	const int validStep = 128;
	const int earlyStop = 1600;
	int stopStep = 0;
	float lastLoss = FLT_MAX;
	int count = 0;
//	std::cout << "conv0 weight shape " << 	net.conv0->weight.sizes() << std::endl;
	ReaderWrapper<DbDefs> reader(dbPath, batchSize, len, batchSize * 4);
	DataType validSet = reader.getValidSet(sampleNum + 1000, batchSize * 2);
	reader.reset();

	std::thread thread(runReader<DbDefs>, std::ref(reader));

	while (count <= epoch) {
		totalNum = 0;
		totalLoss = 0;
		bool printed = false;
		reader.reset();
		while (totalNum < sampleNum) {
			std::vector<Tensor> inputs;
			std::vector<Tensor> labels;

			std::tie(inputs, labels) = reader.read();
			shuffleInput(inputs, labels);
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
			auto loss = torch::nll_loss(output, labelTensor);
			totalLoss += (loss.item<float>() * inputs.size());
//			std::cout << "End of loss " << std::endl;
			auto accu = evaluation(output, labelTensor, len);
			std::cout << "====================================================> loss " << loss.item<float>()
				<< ", " << (totalLoss / totalNum)
				<< ", " << accu << std::endl;


			loss.backward();
			optimizer.step();
		}
//		if ((count % validStep) == 0) {
//			printConvWeight(count, net);
//
//			auto validLoss = validOverfit(reader, net, len, (sampleNum + 10));
//			std::cout << "---------------------------------> Validation loss: " << std::get<0>(validLoss)
//					<< ", " << std::get<1>(validLoss) << std::endl;
//		}

		//TODO: Validation data may not be reusable
		auto validLoss = validOverfitThread(validSet, net, len);
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
//		writeConvWeight(net.conv0->weight);
	}

//	std::cout << "conv0 weights " << std::endl;
//	std::cout << net.conv0->weight << std::endl;

//	printConvWeight(epoch, net);
	reader.stop();
	thread.join();
}

//template<typename NetType, typename DbDefs>
//void testShuffle(LmdbSceneReader<DbDefs>& reader) {
//	const int seqLen = 8;
//	const int batchSize = 8;
//
//	std::vector<Tensor> inputs;
//	std::vector<Tensor> labels;
//	std::map<void*, long> records;
//	std::tie(inputs, labels) = reader.next(batchSize, seqLen);
//}


void getAveLen(LmdbSceneReader<LmdbDataDefs>& reader, LstmNet& net) {
	while (reader.hasNext()) {
		std::vector<Tensor> inputs;
		std::vector<Tensor> labels;

		std::tie(inputs, labels) = reader.next(8);
		net.forward(inputs, true);
	}

	std::cout << "Total length: " << net.totalLen << " samples " << net.totalSample << " ave = "
			<< (double)net.totalLen / net.totalSample << std::endl;
}

void testPad() {
	Tensor tensor = torch::ones({2, 2, 2});
	std::cout << tensor << std::endl;

	tensor = torch::constant_pad_nd(tensor, {0, 0, 0, 0, 0, 1}, 0);
	std::cout << "after " << std::endl;
	std::cout << tensor << std::endl;
}

void testDb(LmdbSceneReader<LmdbDataDefs>& db) {
	Tensor a;
	Tensor b;
	int count = 0;

	while (db.hasNext()) {
		std::tie(a, b) = db.next();
		count ++;

		std::cout << "Read datas " << std::endl << a << std::endl;
		std::cout << "Read labels " << std::endl << b << std::endl;
	}

	std::cout << "Read records: " << count << std::endl;
}

void test5Rows(const float lr) {
	//	dataFile.open("./conv0output.txt");
//		const std::string cppCreateDb = "/home/zf/workspaces/res/dbs/lmdbscenetest";
		const std::string cppCreateDb = "/home/zf/workspaces/res/dbs/lmdb5rowscenetest";
		LmdbSceneReader<LmdbDataDefs> reader(cppCreateDb);
		std::cout << "End of reader construction " << std::endl;
//		LstmNet net;
//		FcNet net;
		FixedFcNet net;
//		PureFcNet net;
//		torch::optim::Adam optimizer (net.parameters(), torch::optim::AdamOptions(lr));
		torch::optim::Adagrad optimizer(net.parameters(), torch::optim::AdagradOptions(lr));

	//	train(reader, net);
	//	std::cout << "------------------------------------------------------------------->" << std::endl;
	//	train(reader, net, 8, optimizer);
	//	testPad();
	//	getAveLen(reader, net);
	//	testDb(reader);

	//	trainSameLen(reader, net, 8, optimizer);
		trainOverfit(reader, net, optimizer);
}

void test5RowThread(const float lr) {
	const std::string cppCreateDb = "/home/zf/workspaces/res/dbs/lmdb5rowscenetest";
	FixedFcNet net;
	torch::optim::Adam optimizer (net.parameters(), torch::optim::AdamOptions(lr));

	trainOverfitThread<FixedFcNet, LmdbDataDefs>(cppCreateDb, net, optimizer);
}

void test2Rows(const float lr) {
	const std::string cpp2RowCreateDb = "/home/zf/workspaces/res/dbs/lmdbscene2rowtest";
	LmdbSceneReader<Lmdb2RowDataDefs> denseReader(cpp2RowCreateDb);

	FcNet net;
//	torch::optim::Adam optimizer (net.parameters(), torch::optim::AdamOptions(lr));
	torch::optim::SGD optimizer (
				net.parameters(), torch::optim::SGDOptions(lr).momentum(0.5));

	trainOverfit(denseReader, net, optimizer);
}

//void testShuffle() {
//	std::vector<int> inputs;
//	std::vector<int> labels;
//
//	for (int i = 0; i < 10; i ++) {
//		inputs.push_back(i);
//		labels.push_back(i);
//	}
//
//	std::cout << "Before " << std::endl;
//	for (int i = 0; i < inputs.size(); i ++) {
//		std::cout << "" << inputs[i] << " --> " << labels[i] << std::endl;
//	}
//
//	shuffleInput(inputs, labels);
//
//	std::cout << std::endl << std::endl << "After " << std::endl;
//	for (int i = 0; i < inputs.size(); i ++) {
//		std::cout << "" << inputs[i] << " --> " << labels[i] << std::endl;
//	}
//}

int main(int argc, char** argv) {
	const float lr = atof(argv[1]);

//	test2Rows(lr);
	test5Rows(lr);
//	testShuffle();
//	test5RowThread(lr);
}
