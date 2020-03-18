/*
 * TextGenerator.h
 *
 *  Created on: Nov 21, 2019
 *      Author: zf
 */

#ifndef INCLUDE_TEXTGENERATOR_H_
#define INCLUDE_TEXTGENERATOR_H_

#include <map>
#include <vector>
#include <torch/torch.h>

struct Net: torch::nn::Module {
//	torch::nn::Embedding embed;
	torch::nn::LSTM lstm;
	torch::nn::LSTM lstm1;
//	torch::nn::GRU lstm;
	torch::nn::Linear fc;
	bool singleStep;

	Net(const int inputLen, const int hiddenLen, bool isSingleStep = false);
	torch::Tensor forward(torch::Tensor input, const int seqLen, const int batchSize);
	torch::Tensor forward(torch::Tensor input, torch::Tensor& state, torch::Tensor& state1, const int seqLen, const int batchSize);
	torch::Tensor forwardSingleStep(torch::Tensor input, const int seqLen, const int batchSize);

};

class Shakespeare {
private:
	const std::string fileName;
	const int hiddenLen;
	Net* net;
	int inputLen;
	std::string text;
	std::vector<char> vocab;
	std::map<char, int> c2i;
	std::map<int, char> i2c;

	std::vector<int> startPos;
	std::vector<int> curPos;
	int sizePerBatch;

	std::pair<torch::Tensor, torch::Tensor> generateInput(int& pos, const int exampleNum);
	std::pair<torch::Tensor, torch::Tensor> generateInput(const int batchSize, const int seqLen, bool test);

	void printText(torch::Tensor output, torch::Tensor target, const int seqLen, const int batchSize) ;
	void resetPos(const int batchSize);

public:
	Shakespeare (std::string filePath, int hiddenLen);
	~Shakespeare();
	void init();
	void train(const int epochNum, const int seqLen, const int batchSize, const int totalExampleNum);
	void generateText(const int textLen, const int candidateNum);
};



#endif /* INCLUDE_TEXTGENERATOR_H_ */
