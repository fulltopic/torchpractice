/*
 * cnnnet.h
 *
 *  Created on: Mar 16, 2020
 *      Author: zf
 */

#ifndef INCLUDE_NETS_CNNNET_H_
#define INCLUDE_NETS_CNNNET_H_

#include "torch/torch.h"
#include <vector>

#include <torch/torch.h>
#include <fstream>
#include <iostream>
#include <string>


class LayerConf {
public:
	virtual ~LayerConf();
	virtual torch::IntArrayRef outputDim() = 0;
};


class ConvLayerConf: public LayerConf {
public:
	virtual ~ConvLayerConf() = default;
	ConvLayerConf();
//	ConvLayerConf& ConvLayerConf(const ConvLayerConf& copied) = default;

private:
	int kernelH;
	int kernelW;
	int padH;
	int padW;
	int strideH;
	int strideW;
	bool pooled;
	int poolH;
	int poolW;

public:
	//TODO: Combine h/w
	inline ConvLayerConf& setKernelH(const int h) {
		kernelH = h;
		return *this;
	}
	ConvLayerConf& setKernelW(const int w);
	ConvLayerConf& setPadH(const int h);
	ConvLayerConf& setPadW(const int w);
	ConvLayerConf& setStrideH(const int h);
	ConvLayerConf& setStrideW(const int w);
	ConvLayerConf& setPoolH(const int h);
	ConvLayerConf& setPoolW(const int w);

	inline int getKernelH() {
		return kernelH;
	}

	virtual torch::IntArrayRef outputDim();
};

class BatchNormLayerConf: public LayerConf {
public:
	virtual ~BatchNormLayerConf() = default;
	BatchNormLayerConf(const int batchChan);

private:
	const int batchNum;

public:
	inline int getBatchChan() {
		return batchNum;
	}

	virtual torch::IntArrayRef outputDim();
};

class DenseLayerConf: public LayerConf {
public:
	virtual ~DenseLayerConf() = default;
	DenseLayerConf(const int inputNum, const int outputNum);

private:
	const int input;
	const int output;

public:
	inline int getInput() { return input; }
	inline int getOutput() { return output; }

	virtual torch::IntArrayRef outputDim();
};

class NetworkConf {
private:
	std::vector<std::unique_ptr<LayerConf>> layers;

public:
	NetworkConf() = default;
	~NetworkConf() = default;

	void addLayer(std::unique_ptr<LayerConf> layer);

};


struct CNNNet: torch::nn::Module {
	torch::nn::Conv2d conv0;
	torch::nn::BatchNorm batchNorm0;
	torch::nn::Conv2d conv1;
	torch::nn::BatchNorm batchNorm1;
	torch::nn::Linear fc0;
	torch::nn::BatchNorm fcBatchNorm0;
	torch::nn::Linear fc1;

	std::ofstream dataFile;

	CNNNet();
	~CNNNet();

	void setTrain(bool isTrain);
	torch::Tensor forward(std::vector<torch::Tensor> inputs, const int seqLen, bool isTrain = true, bool toRecord = false);
	torch::Tensor inputPreprocess(torch::Tensor);

	static const std::string GetName();
};


#endif /* INCLUDE_NETS_CNNNET_H_ */
