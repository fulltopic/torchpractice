/*
 * syncplotserver.h
 *
 *  Created on: Mar 9, 2020
 *      Author: zf
 */

#ifndef INCLUDE_PYTOOLS_SYNCPLOTSERVER_H_
#define INCLUDE_PYTOOLS_SYNCPLOTSERVER_H_

#include <vector>
#include <queue>
#include <atomic>
#include <time.h>
#include <condition_variable>
#include <mutex>

#include <torch/torch.h>


class SyncPlotServer {
public:
	const static int DataCap = 1024;
	const static int RowNum = 3;
	const static int ColNum = 2;
	const static int SubPlotNum = RowNum * ColNum;

private:
	const std::string figureName;

	int trainIte;
	uint64_t trainSeq;
	float currTrainLoss;
	float currTrainAccu;
	std::vector<float> trainLoss;
	std::vector<float> trainAveLoss;
	std::vector<float> trainAccu;
	std::vector<float> trainAveAccu;

	int validIte;
	std::vector<float> validLoss;
	std::vector<float>  validAccu;

	const int updateRatioNum;
	std::vector<torch::Tensor> lastParams;
	std::vector<std::vector<float>> updateRatio;

public:
	SyncPlotServer(const int paramNum, const std::vector<torch::Tensor>& parameters, const std::string iFigureName = "");
	~SyncPlotServer() = default;
	SyncPlotServer(const SyncPlotServer& a) = delete;
	SyncPlotServer& operator=(SyncPlotServer& a) = delete;

	void trainUpdate(const torch::Tensor outputs, const torch::Tensor labels, const std::vector<torch::Tensor> parameters);
	void validUpdate(const torch::Tensor outputs, const torch::Tensor labels);
	void refresh();
	void save(const std::string fileName);

private:
	void adjustTrainVec();
	void adjustValidVec();
	float getAccu(const torch::Tensor outputs, const torch::Tensor labels);
};


#endif /* INCLUDE_PYTOOLS_SYNCPLOTSERVER_H_ */
