/*
 * syncplotserver.cpp
 *
 *  Created on: Mar 9, 2020
 *      Author: zf
 */

#include "pytools/syncplotserver.h"
//#include "pytools/batchplotserver.h"
#include <matplotlibcpp.h>
#include <iostream>
#include <time.h>
#include <thread>
#include <cmath>
#include <torch/torch.h>

using std::vector;
using torch::Tensor;
using std::string;
using std::cout;
using std::endl;

SyncPlotServer::SyncPlotServer(const int paramNum, const std::vector<Tensor>& parameters, const std::string iFigureName, const bool toPlotValidTrain):
		figureName(iFigureName),
		trainIte(0),
		trainSeq(0),
		currTrainLoss(0),
		currTrainAccu(0),
		trainLoss(DataCap, 0),
		trainAveLoss(DataCap, 0),
		trainAccu(DataCap, 0),
		trainAveAccu(DataCap, 0),
		validIte(0),
		validLoss(DataCap, 0),
		validAccu(DataCap, 0),
		validTrainAveLoss(DataCap, 0),
		validTrainMinLoss(DataCap, 0),
		validTrainMaxAccu(DataCap, 0),
		validTrainAveAccu(DataCap, 0),
		updateRatioNum(paramNum),
		lastParams(vector<Tensor>()),
		updateRatio(paramNum, vector<float>(DataCap, 0)),
		plotValidTrain(toPlotValidTrain)
{
//	matplotlibcpp::title(figureName);
	for (int i = 0; i < paramNum; i ++) {
		lastParams.push_back(parameters[i].clone());
	}
}

void SyncPlotServer::adjustTrainVec() {
	for (int i = 0; i < DataCap / 2; i ++) {
		trainLoss[i] = trainLoss[i * 2];
		trainAveLoss[i] = trainAveLoss[i * 2];
		trainAccu[i] = trainAccu[i * 2];
		trainAveAccu[i] = trainAveAccu[i * 2];

		for (int j = 0; j < updateRatioNum; j ++) {
			updateRatio[j][i] = updateRatio[j][i * 2];
		}
	}

	for (int i = DataCap / 2; i < DataCap; i ++) {
		trainLoss[i] = 0;
		trainAveLoss[i] = 0;
		trainAccu[i] = 0;
		trainAveAccu[i] = 0;

		for (int j = 0; j < updateRatioNum; j ++) {
			updateRatio[j][i] = 0;
		}
	}

	trainIte = DataCap / 2;
}

void SyncPlotServer::adjustValidVec() {
	for (int i = 0; i < DataCap / 2; i ++) {
		validLoss[i] = validLoss[i * 2];
		validAccu[i] = validAccu[i * 2];
		validTrainMinLoss[i] = validTrainMinLoss[i * 2];
		validTrainMaxAccu[i] = validTrainMaxAccu[i * 2];
		validTrainAveLoss[i] = validTrainAveLoss[i * 2];
		validTrainAveAccu[i] = validTrainAveAccu[i * 2];
	}

	for (int i = DataCap / 2; i < DataCap; i ++) {
		validLoss[i] = 0;
		validAccu[i] = 0;
		validTrainMinLoss[i] = 0;
		validTrainMaxAccu[i] = 0;
		validTrainAveLoss[i] = 0;
		validTrainAveAccu[i] = 0;
	}

	validIte = DataCap / 2;
}

float SyncPlotServer::getAccu(const Tensor outputs, const Tensor labels) {
	Tensor indices = torch::argmax(outputs, 1);
	Tensor diff = torch::sub(labels, indices);
	int total = labels.size(0);
	int matched = total - diff.nonzero().size(0);

	return (float)matched / total;
}

void SyncPlotServer::trainUpdate(const Tensor outputs, const Tensor labels, const vector<Tensor> parameters) {
	trainSeq ++;

	if (trainIte >= DataCap) {
		adjustTrainVec();
	}

	Tensor lossTensor = torch::nll_loss(outputs, labels);
	float loss = lossTensor.item<float>();
	float accu = getAccu(outputs, labels);
	trainLoss[trainIte] = loss;
	trainAccu[trainIte] = accu;

	currTrainLoss = ((double)currTrainLoss * (trainSeq - 1) + loss) / (trainSeq);
	currTrainAccu = ((double)currTrainAccu * (trainSeq - 1) + accu) / (trainSeq);
	trainAveLoss[trainIte] = currTrainLoss;
	trainAveAccu[trainIte] = currTrainAccu;

	for (int i = 0; i < updateRatioNum; i ++) {
		Tensor updateWs = parameters[i].sub(lastParams[i]);
		Tensor ratioTensor = updateWs.norm().div(lastParams[i].norm());
		float ratio = ratioTensor.item<float>();
		updateRatio[i][trainIte] = log10(ratio);
		lastParams[i] = parameters[i].clone();
	}

	trainIte ++;

//	refresh();
}

void SyncPlotServer::validUpdate(const Tensor outputs, const Tensor labels, vector<float> params) {
	if (validIte >= DataCap) {
		adjustValidVec();
	}

	Tensor lossTensor = torch::nll_loss(outputs, labels);
	validLoss[validIte] = lossTensor.item<float>();
	validAccu[validIte] = getAccu(outputs, labels);

	if (plotValidTrain) {
		validTrainMinLoss[validIte] = params[0];
		validTrainAveLoss[validIte] = params[1];
		validTrainMaxAccu[validIte] = params[2];
		validTrainAveAccu[validIte] = params[3];
	}
	validIte ++;

//	refresh();
}

void SyncPlotServer::refresh() {
	matplotlibcpp::clf();
	int extraSub = 0;
	int extraRow = 0;
	if (plotValidTrain) {
		extraSub = 2;
		extraRow = 1;
	}

	matplotlibcpp::subplot(RowNum + extraRow, ColNum, 1);
	matplotlibcpp::grid(true);
	matplotlibcpp::title("TrainLoss");
	matplotlibcpp::plot(std::vector<float>(&trainLoss[0], &trainLoss[trainIte]));
	matplotlibcpp::plot(std::vector<float>(&trainAveLoss[0], &trainAveLoss[trainIte]));
	matplotlibcpp::subplot(RowNum + extraRow, ColNum, 2);
	matplotlibcpp::grid(true);
	matplotlibcpp::title("TrainAccuracy");
	matplotlibcpp::plot(std::vector<float>(&trainAccu[0], &trainAccu[trainIte]));
	matplotlibcpp::plot(std::vector<float>(&trainAveAccu[0], &trainAveAccu[trainIte]));

	if (plotValidTrain) {
	matplotlibcpp::subplot(RowNum + extraRow, ColNum, 3);
	matplotlibcpp::grid(true);
	matplotlibcpp::title("ValidTrainLoss");
	matplotlibcpp::plot(std::vector<float>(&validTrainMinLoss[0], &validTrainMinLoss[validIte]));
	matplotlibcpp::plot(std::vector<float>(&validTrainAveLoss[0], &validTrainAveLoss[validIte]));
	matplotlibcpp::subplot(RowNum + extraRow, ColNum, 4);
	matplotlibcpp::grid(true);
	matplotlibcpp::title("ValidTrainAccuracy");
	matplotlibcpp::plot(std::vector<float>(&validTrainMaxAccu[0], &validTrainMaxAccu[validIte]));
	matplotlibcpp::plot(std::vector<float>(&validTrainAveAccu[0], &validTrainAveAccu[validIte]));
	}


	matplotlibcpp::subplot(RowNum + extraRow, ColNum, 3 + extraSub);
	matplotlibcpp::grid(true);
	matplotlibcpp::title("ValidLoss");
	matplotlibcpp::plot(std::vector<float>(&validLoss[0], &validLoss[validIte]));
	matplotlibcpp::subplot(RowNum + extraRow, ColNum, 4 + extraSub);
	matplotlibcpp::grid(true);
	matplotlibcpp::title("ValidAccuracy");
	matplotlibcpp::plot(std::vector<float>(&validAccu[0], &validAccu[validIte]));

	matplotlibcpp::subplot(RowNum + extraRow, ColNum, 5 + extraSub);
	matplotlibcpp::grid(true);
	matplotlibcpp::title("UpdateRatio(half-)");
	for(int i = 0; i < updateRatioNum / 2; i ++) {
//		if ((i % 2) == 0) {
			matplotlibcpp::plot(std::vector<float>(&updateRatio[i][0], &updateRatio[i][trainIte]));
//		}
	}
//	matplotlibcpp::plot(std::vector<float>(&updateRatio[0][0], &updateRatio[0][trainIte]), "b");
//	matplotlibcpp::plot(std::vector<float>(&updateRatio[1][0], &updateRatio[1][trainIte]), "r");
//	matplotlibcpp::plot(std::vector<float>(&updateRatio[2][0], &updateRatio[2][trainIte]), "g");
//	matplotlibcpp::plot(std::vector<float>(&updateRatio[3][0], &updateRatio[3][trainIte]), "c");
//	matplotlibcpp::plot(std::vector<float>(&updateRatio[4][0], &updateRatio[4][trainIte]), "m");
//	matplotlibcpp::plot(std::vector<float>(&updateRatio[5][0], &updateRatio[5][trainIte]), "y");
//	matplotlibcpp::plot(std::vector<float>(&updateRatio[6][0], &updateRatio[6][trainIte]), "k");
//	matplotlibcpp::plot(std::vector<float>(&updateRatio[7][0], &updateRatio[7][trainIte]), "tab:purple");
//	matplotlibcpp::plot(std::vector<float>(&updateRatio[8][0], &updateRatio[8][trainIte]), "tab:blue");

	matplotlibcpp::subplot(RowNum + extraRow, ColNum, 6 + extraSub);
	matplotlibcpp::grid(true);
	matplotlibcpp::title("UpdateRatio(half1)");
	for(int i = updateRatioNum / 2; i < updateRatioNum; i ++) {
//		if ((i % 2) == 1) {
			matplotlibcpp::plot(std::vector<float>(&updateRatio[i][0], &updateRatio[i][trainIte]));
//		}
	}
//	matplotlibcpp::plot(std::vector<float>(&updateRatio[9][0], &updateRatio[9][trainIte]), "tab:gray");
//	matplotlibcpp::plot(std::vector<float>(&updateRatio[10][0], &updateRatio[10][trainIte]), "tab:pink");
//	matplotlibcpp::plot(std::vector<float>(&updateRatio[11][0], &updateRatio[11][trainIte]), "b");
//	matplotlibcpp::plot(std::vector<float>(&updateRatio[12][0], &updateRatio[12][trainIte]), "r");
//	matplotlibcpp::plot(std::vector<float>(&updateRatio[13][0], &updateRatio[13][trainIte]), "g");
//	matplotlibcpp::plot(std::vector<float>(&updateRatio[14][0], &updateRatio[14][trainIte]), "c");
//	matplotlibcpp::plot(std::vector<float>(&updateRatio[15][0], &updateRatio[15][trainIte]), "m");
//	matplotlibcpp::plot(std::vector<float>(&updateRatio[16][0], &updateRatio[16][trainIte]), "y");
//	matplotlibcpp::plot(std::vector<float>(&updateRatio[17][0], &updateRatio[17][trainIte]), "k");
//	matplotlibcpp::plot(std::vector<float>(&updateRatio[18][0], &updateRatio[18][trainIte]), "tab:purple");
//	matplotlibcpp::plot(std::vector<float>(&updateRatio[19][0], &updateRatio[19][trainIte]), "tab:blue");
//	matplotlibcpp::plot(std::vector<float>(&updateRatio[20][0], &updateRatio[20][trainIte]), "tab:gray");
//	matplotlibcpp::plot(std::vector<float>(&updateRatio[21][0], &updateRatio[21][trainIte]), "tab:pink");

	matplotlibcpp::pause(0.01);
//	cout << "End of refresh" << endl;
}

void SyncPlotServer::save(const string fileName) {
	matplotlibcpp::figure_size(2000, 2500);
	refresh();
	matplotlibcpp::save(fileName);
}
