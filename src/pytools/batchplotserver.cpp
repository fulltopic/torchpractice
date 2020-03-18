/*
 * batchplotserver.cpp
 *
 *  Created on: Mar 3, 2020
 *      Author: zf
 */

#include "pytools/batchplotserver.h"
#include <matplotlibcpp.h>
#include <iostream>
#include <time.h>
#include <thread>
#include <cmath>


BatchPlotServer::BatchPlotServer():
				datas(PlotNum, std::vector<float>(Cap, 0.0f)),
				indexes(PlotNum, 0),
				q(QCap, std::make_pair(0, 0.0f)),
				toRead(false),
				pushIndex(0),
				waitT{WaitSec, WaitMSec},
				running(true)
{
}

void BatchPlotServer::newEvent(std::pair<int, float> data) {
//	std::cout << "newEvent " << pushIndex << std::endl;
	while (pushIndex >= QCap) {
		notifyRead();

		std::unique_lock<std::mutex> lock(pushMutex);
		pushCv.wait(lock);
	}

//	std::cout << "push event " << pushIndex << std::endl;
	q[pushIndex] = data;
	pushIndex ++;
}

void BatchPlotServer::notifyRead() {
	std::unique_lock<std::mutex> lock(popMutex);
	toRead = true;
	popCv.notify_one();
}


void BatchPlotServer::threadMain() {
//	std::cout << "Plotserver constructor " << std::endl;
	matplotlibcpp::figure_size(2000, 2000);
	matplotlibcpp::suptitle("Test");
//	std::cout << "End of plotserver constructor " << std::endl;

	struct timespec rem;

	while (running) {
//		std::cout << "threadMain " << std::endl;
		if (toRead) {
//			std::cout << "Read event " << std::endl;
//			std::cout << "To pop queue " << std::endl;
			for (int i = 0; i < pushIndex; i ++) {
				newData(q[i].first, q[i].second);
			}

//			std::cout << "Read data " << data.first << ", " << data.second << std::endl;
//			std::cout << "To refresh " << std::endl;
			refresh();
			pushIndex = 0;
			toRead = false;

			std::unique_lock<std::mutex> lock(pushMutex);
			pushCv.notify_one();
		} else {
//			std::cout << "Sleep " << std::endl;
//			nanosleep(&waitT, &rem);
//			std::cout << "End of sleep" << std::endl;
			std::unique_lock<std::mutex> lock(popMutex);
			popCv.wait(lock);
		}
	}
}

void BatchPlotServer::newData(int index, float data) {
	int plotIndex = index - 1;
//	std::cout << "plotIndex = " << plotIndex << std::endl;
	if (plotIndex >= PlotNum || plotIndex < 0) {
		return;
	}

	int vecIndex = indexes[plotIndex];
//	std::cout << "vecIndex = " << vecIndex << std::endl;
	if (vecIndex >= Cap) {
		for (int i = 0; i < Cap; i ++) {
			datas[plotIndex][i] = datas[plotIndex][i * 2];
		}
		for (int i = Cap / 2; i < Cap; i ++) {
			datas[plotIndex][i] = 0;
		}
		vecIndex = Cap / 2; //Cap % 2 == 0

	}

	datas[plotIndex][vecIndex] = data;
	indexes[plotIndex] = (vecIndex + 1);
//	std::cout << "End of newData " << plotIndex << ", " << indexes[plotIndex] << std::endl;
}

void BatchPlotServer::refresh() {
//	matplotlibcpp::subplot(PlotW, PlotH, index);
//	std::cout << "subplot " << index << std::endl;
////	matplotlibcpp::clf();
//	matplotlibcpp::plot(datas[index - 1]);
//	std::cout << "Ploted " << std::endl;
////	matplotlibcpp::pause(0.01);
////	std::cout << "End of pause " << std::endl;
//	matplotlibcpp::draw();
//	std::vector<int> indexVec(Cap, 0);
//	for (int i = 0; i < Cap; i ++) {
//		indexVec[i] = i;
//	}

	matplotlibcpp::clf();
	matplotlibcpp::grid(true);
//	matplotlibcpp::plot(datas[0]);
//	std::cout << "Get subvector " << indexes[0] << std::endl;
//	int index0 = indexes[0];
//	std::vector<int> testIndex(&indexVec[0], &indexVec[index0]);
//	std::vector<float> testData(&datas[0][0], &datas[0][index0]);
//	for (int i = 0; i < index0; i ++) {
//		std::cout << "Scatter data " << testIndex[i] << ", " << testData[i] << std::endl;
//	}
//	matplotlibcpp::scatter(testIndex, testData);
	matplotlibcpp::subplot(2, 2, 1);
	matplotlibcpp::grid(true);
	matplotlibcpp::plot(std::vector<float>(&datas[0][0], &datas[0][indexes[0]]));
	matplotlibcpp::subplot(2, 2, 2);
	matplotlibcpp::grid(true);
//	matplotlibcpp::plot(datas[1]);
	matplotlibcpp::plot(std::vector<float>(&datas[1][0], &datas[1][indexes[1]]));
	matplotlibcpp::subplot(2, 2, 3);
//	matplotlibcpp::plot(datas[2]);
	matplotlibcpp::plot(std::vector<float>(&datas[2][0], &datas[2][indexes[2]]));
	matplotlibcpp::subplot(2, 2, 4);
//	matplotlibcpp::plot(datas[3]);
	matplotlibcpp::plot(std::vector<float>(&datas[3][0], &datas[3][indexes[3]]));
	matplotlibcpp::pause(0.01);
}

BatchPlotServer& BatchPlotServer::GetInstance() {
	static BatchPlotServer instance;

	return instance;
}

void BatchPlotServer::Run() {
	BatchPlotServer& instance = GetInstance();
	std::cout << "Run get instance " << std::endl;
	instance.threadMain();
}

void BatchPlotServer::test() {
	std::vector<std::vector<float>> testData(4, std::vector<float>());
//	std::vector<float> test(10, 0.5f);


//	matplotlibcpp::plot(test);
//	matplotlibcpp::show();

	for (int i = 0; i < 20; i ++) {
		for (int j = 0; j < 4; j ++) {
			testData[j].push_back(i);
		}
		matplotlibcpp::subplot(2, 2, 1);
		matplotlibcpp::plot(testData[0]);
		matplotlibcpp::subplot(2, 2, 2);
		matplotlibcpp::plot(testData[1]);
		matplotlibcpp::subplot(2, 2, 3);
		matplotlibcpp::plot(testData[2]);
		matplotlibcpp::subplot(2, 2, 4);
		matplotlibcpp::plot(testData[3]);
		matplotlibcpp::pause(0.01);
	}

//	int n = 500;
//	std::vector<double> x(n), y(n), z(n), w(n,2);
//	for(int i=0; i<n; ++i) {
//		x.at(i) = i;
//		y.at(i) = sin(2*M_PI*i/360.0);
//		z.at(i) = 100.0 / i;
//	}
//
//    // Set the "super title"
//	matplotlibcpp::suptitle("My plot");
//	matplotlibcpp::subplot(1, 2, 1);
//	matplotlibcpp::plot(x, y, "r-");
//	matplotlibcpp::subplot(1, 2, 2);
//	matplotlibcpp::plot(x, z, "k-");
//    // Add some text to the plot
//	matplotlibcpp::text(100, 90, "Hello!");
//
//
//	// Show plots
//	matplotlibcpp::show();
}
