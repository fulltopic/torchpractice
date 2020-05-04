#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>

#include <torch/torch.h>

using namespace std;
using namespace torch;
using namespace torch::nn;

//const string trainFileName = "./data/shakespeare.txt";

void inputPreprocess(const string trainFileName, string& text,
		vector<char>& vocab, map<char, int>& char_to_idx, map<int, char>& idx_to_char) {
	  ifstream infile(trainFileName);
	  stringstream buffer;
	  buffer << infile.rdbuf();
	  text = buffer.str();

	  if (!text.size()) {
	    cerr << "unable to read input text" << endl;
	    return;
	  }

	  set<char> vocab_set(text.begin(), text.end());
	  vocab.assign(vocab_set.begin(), vocab_set.end());

	  auto index = 0;
	  for (auto c : vocab) {
	    char_to_idx[c] = index;
	    idx_to_char[index++] = c;
	  }

	  cout << "Input has " << vocab.size()
	            << " characters. Total input size: " << text.size() << endl;
}

void generateDataFile(const string inputFileName, const string saveFileName, const string mapFileName, const int exampleNum) {
	vector<char> vocab;
	map<char, int> char2Index;
	map<int, char> index2Char;
	string text;

	inputPreprocess(inputFileName, text, vocab, char2Index, index2Char);

	const int64_t inputLen = vocab.size();
	const int64_t actualExampleNum = std::min<int64_t>(exampleNum, text.length() - 1);
	Tensor exampleTensor = torch::zeros({actualExampleNum, (inputLen + 1)});
	auto examples = torch::chunk(exampleTensor, exampleTensor.size(0), 0);
	auto data = exampleTensor.data_ptr<float>();

	for (int i = 0; i < exampleNum && i < (text.length() - 1); i ++) {
		data[exampleNum * exampleTensor.size(1) + char2Index[text.at(i)]] = 1.0;
		data[exampleNum * exampleTensor.size(1) + inputLen] = text.at(i + 1);
	}

	torch::save(exampleTensor, saveFileName);

	int64_t mapSize = static_cast<int64_t>(char2Index.size());
	Tensor char2IdxTensor = torch::zeros({mapSize * 2, 2}).to(ScalarType::Long);
	auto dataPtr = char2IdxTensor.data_ptr<int64_t>();
	int index = 0;
	for (auto pair: char2Index) {
		dataPtr[index * 2] = pair.first;
		dataPtr[index * 2 + 1] = pair.second;
		index ++;
	}
	cout << "Index 2 char map " << endl;
	for (auto pair: index2Char) {
		dataPtr[index * 2] = pair.first;
		dataPtr[index * 2 + 1] = pair.second;
		cout << dataPtr[index * 2] << " --> " << dataPtr[index * 2 + 1] << endl;
		index ++;
	}
	torch::save(char2IdxTensor, mapFileName);
}

pair<Tensor, Tensor> generateInputTensor(const string& text, int& pos,
		vector<char> vocab, map<char, int> c2i, map<int, char> i2c, const int exampleNum) {
	const int64_t inputLen = vocab.size();
	Tensor inputTensor = torch::zeros({exampleNum, inputLen});
	Tensor targetTensor = torch::zeros({exampleNum, 1});

	auto inputData = (float*)inputTensor.data_ptr();
	auto targetData = (float*)targetTensor.toType(ScalarType::Long).data_ptr();

	for (int i = 0; i < exampleNum; i ++) {
		char c = text.at(pos);
		pos ++;

		inputData[inputLen * i + c2i[c]] = 1;
		targetData[i] = c2i[text.at(pos + 1)];
	}

	return make_pair(inputTensor, targetTensor);
}

void readDataFile(const string dataFileName) {
	Tensor tensor;
	torch::load(tensor, dataFileName);
	cout << tensor.sizes() << endl;
}

struct Net: torch::nn::Module {
	LSTM lstm;
	Linear fc;
//	int batchSize;
//	int seqLen;


//	Net(int seq, int hiddenLen
//			, int inputLen, int batch
//			):
	Net(int hiddenLen, int inputLen):
		lstm(LSTMOptions(inputLen, hiddenLen)),
		fc(hiddenLen, inputLen)
//		batchSize(batch),
//		seqLen(seq)
	{
//		batchSize = batch;
//		seqLen = seq;
		register_module("lstm", lstm);
		register_module("fc", fc);
	}

	Tensor forward(Tensor input, const int seqLen, const int batchSize) {
		Tensor state0 = torch::zeros({1, input.size(0), lstm->options.hidden_size()});
		Tensor state1 = torch::zeros({1, input.size(0), lstm->options.hidden_size()});
		std::tuple<Tensor, Tensor> state(state0, state1);

		auto inputs = torch::chunk(input, input.size(0) / (seqLen * batchSize), 0);
//		auto test = inputs[0].view({seqLen, batchSize, input.size(1)});
//		cout << test.sizes() << endl;
//		cout << "inputs " << input.sizes() << ", " << input.numel() << endl;
		vector<Tensor> outputs;
		for (auto inputTensor: inputs) {
			auto lstmOutput = lstm->forward(inputTensor.view({seqLen, batchSize, input.size(input.dim() - 1)}), state);
			state = std::get<1>(lstmOutput);
			auto fcOutput = fc->forward(std::get<0>(lstmOutput));
			auto output = torch::softmax(fcOutput, 2);
//			auto output = torch::log_softmax(fcOutput, 2);
//			cout << "output " << output.sizes() << endl;
			outputs.push_back(output.view({seqLen * batchSize, input.size(input.dim() - 1)}));
		}

		return torch::cat(outputs, 0);
	}
};

void testLoss() {
	auto input = torch::randn({3, 5}, torch::requires_grad(true));
	auto target = torch::empty(3, torch::kLong).random_(5);
	cout << input.sizes() << ", " << target.sizes() << endl;
	auto output = torch::nll_loss(torch::log_softmax(input, /*dim=*/1), target);
//	output.backward();
}

pair<map<char, int>, map<int, char> > readMapFile(string fileName) {
	Tensor tensor;
	torch::load(tensor, fileName);
	tensor = tensor.toType(ScalarType::Long);
	cout << "Total size " << tensor.size(0) << endl;
	auto data = (int64_t*)tensor.data_ptr();

	map<char, int> ch2Idx;
	map<int, char> idx2Ch;
	int index = 0;
	for (; index < tensor.size(0) / 2; index ++) {
		ch2Idx[data[index * 2]] = data[index * 2 + 1];
	}
	for (; index < tensor.size(0); index ++) {
		idx2Ch[data[index * 2]] = data[index * 2 + 1];
	}

	return make_pair(ch2Idx, idx2Ch);
}

void generateText(Net& net, const int inputLen, const int textLen, const string mapFileName) {
	map<char, int> ch2Idx;
	map<int, char> idx2Char;

	const int maxCandidates = 4;

	tie(ch2Idx, idx2Char) = readMapFile(mapFileName);
	stringstream text;
	int64_t chIdx = rand() % idx2Char.size();
	text << idx2Char[chIdx];

	for (int i = 0; i < textLen; i ++) {
		Tensor input = torch::zeros({1, 1, inputLen});
		auto ptr = input.data_ptr<float>();
		ptr[chIdx] = 1;

//		cout << "input length " << inputLen << endl;
//		cout << input.sizes() << endl;
		auto output = net.forward(input, 1, 1);
//		cout << "output " << output << endl;
		auto maxOutput = output.argmax();
		auto sortOutput = output.argsort(output.dim() - 1, true);
//		cout << "argmax " << maxOutput << endl;
//		cout << "max output " << sortOutput.sizes() << ": " << sortOutput << endl;
		auto maxIndice = (long*)sortOutput.data_ptr();
		chIdx = maxIndice[rand() % maxCandidates];
//		chIdx = (maxOutput.item().toLong());
		text << idx2Char[chIdx];
	}

	cout << "Generated text: " << endl;
	cout << text.str() << endl;
}

void train(const int epochNum, const int hiddenLen, const int seqLen, const int batchSize,
		const string dataFileName, const string mapFileName) {
	Tensor tensor;
	torch::load(tensor, dataFileName);

	const int inputLen = tensor.size(1) - 1;
	Tensor inputTensor = torch::narrow(tensor, 1, 0, inputLen);
	Tensor targetTensor = torch::narrow(tensor, 1, inputLen, 1).to(ScalarType::Long);
//	targetTensor = targetTensor.to(ScalarType::Long);
//	auto optimizer torch::optim::SGD(0.5);

	Net net(hiddenLen, inputLen);//, inputLen, batchSize);
//	torch::optim::Adam optimizer(net.parameters(), optim::AdamOptions(0.5));
	torch::optim::LBFGS optimizer (net.parameters(), torch::optim::LBFGSOptions(0.5));

	auto cost = [&]() {
		optimizer.zero_grad();
		auto output = net.forward(inputTensor, seqLen, batchSize);
		auto loss = torch::nll_loss(output, targetTensor.view({targetTensor.size(0)}));
		loss.backward();

		return loss;
	};

	for (int i = 0; i < epochNum; i ++) {
		auto loss = optimizer.step(cost);
		cout << "loss " << loss << endl;
		generateText(net, inputLen, 40, mapFileName);

	}

//	for (int i = 0; i < epochNum; i ++) {
//		optimizer.zero_grad();
//		Tensor output = net.forward(inputTensor, seqLen, batchSize);
//		cout << output.sizes() << ", " << targetTensor.sizes() << ", " << targetTensor.dim() << endl;
//		Tensor loss = torch::nll_loss(output, targetTensor.view({targetTensor.size(0)}));
//		loss.backward();
//		optimizer.step();
//		cout << "loss " << loss << endl;
//	}

}

int main() {
	const string textFileName = "./data/shakespeare.txt";
	const int exampleNum = 1024 * 1024;
	const string dataFileName = "./data/shakespeare" + std::to_string(exampleNum) + ".pt";
	const string mapFileName = "./data/shakespeare_map.pt";

//	readMapFile(mapFileName);
	generateDataFile(textFileName, dataFileName, mapFileName, exampleNum);
//	readDataFile(dataFileName);
//	train(10, 200, 32, 32, dataFileName, mapFileName);//	testLoss();
}
