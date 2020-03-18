/*
 * plotserver.cpp
 *
 *  Created on: Feb 28, 2020
 *      Author: zf
 */


#include "pytools/plotserver.h"
#include <matplotlibcpp.h>
#include <iostream>
#include <time.h>
#include <thread>
#include <cmath>


PlotServer::PlotServer():
				datas(PlotNum, std::vector<float>(Cap, 0.0f)),
				indexes(PlotNum, 0),
				q(QCap, std::make_pair(0, 0.0f)),
				qPushIndex(1),
				qPopIndex(1),
				qPushTail(0),
				waitT{WaitSec, WaitMSec},
				running(true)
{
}

void PlotServer::newEvent(std::pair<int, float> data) {
	bool done = false;
	while (!done) {
		//TODO: Check limit
		if (qPushIndex - qPopIndex >= QCap) {
			std::cout << "Queue full, data abandoned" << std::endl;
			return;
		}

		std::uint32_t index = qPushIndex;
		done = qPushIndex.compare_exchange_strong(index, index + 1);
		if (done) {
			q[index % QCap] = data;

			while (qPushTail != (index - 1)) {
				nanosleep(&waitT, NULL);
			}
			qPushTail = index;
//			std::cout << "Pushed " << "<" << data.first << "," << data.second << "> into " << index << std::endl;
		}
	}
}

void PlotServer::stop() {
	running = false;
}

void PlotServer::threadMain() {
//	std::cout << "Plotserver constructor " << std::endl;
	matplotlibcpp::figure_size(2000, 2000);
	matplotlibcpp::suptitle("Test");
//	std::cout << "End of plotserver constructor " << std::endl;

	struct timespec rem;

	while (running) {
//		std::cout << "To pop an event " << qPopIndex << ", " << qPushTail << std::endl;
		if (qPopIndex <= qPushTail) {
//			std::cout << "Read event " << std::endl;
			uint32_t index = qPopIndex;
			std::pair<int, float> data = q[(int)(index % QCap)];
			qPopIndex ++;

//			std::cout << "Read data " << data.first << ", " << data.second << std::endl;
			newData(data.first, data.second);
			refresh(data.first);
		} else {
//			std::cout << "Sleep " << std::endl;
			nanosleep(&waitT, &rem);
//			std::cout << "End of sleep" << std::endl;
		}
	}
}

void PlotServer::newData(int index, float data) {
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
	indexes[plotIndex] = vecIndex + 1;
//	std::cout << "End of newData " << std::endl;
}

void PlotServer::refresh(int index) {
	matplotlibcpp::subplot(PlotW, PlotH, index);
//	std::cout << "subplot " << index << std::endl;
////	matplotlibcpp::clf();
	matplotlibcpp::plot(datas[index - 1]);
//	std::cout << "Ploted " << std::endl;
////	matplotlibcpp::pause(0.01);
////	std::cout << "End of pause " << std::endl;
//	matplotlibcpp::draw();

//	matplotlibcpp::subplot(2, 2, 1);
//	matplotlibcpp::plot(datas[0]);
//	matplotlibcpp::subplot(2, 2, 2);
//	matplotlibcpp::plot(datas[1]);
//	matplotlibcpp::subplot(2, 2, 3);
//	matplotlibcpp::plot(datas[2]);
//	matplotlibcpp::subplot(2, 2, 4);
//	matplotlibcpp::plot(datas[3]);
	matplotlibcpp::pause(0.01);
}

PlotServer& PlotServer::GetInstance() {
	static PlotServer instance;

	return instance;
}

void PlotServer::Run() {
	PlotServer& instance = GetInstance();
	std::cout << "Run get instance " << std::endl;
	instance.threadMain();
}

void PlotServer::test() {
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


