/*
 * testplot.cpp
 *
 *  Created on: Feb 26, 2020
 *      Author: zf
 */



#include <matplotlibcpp.h>
#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <thread>
#include <cmath>
#include <chrono>
#include <time.h>
#include <unistd.h>

#include <pytools/plotserver.h>

namespace plt = matplotlibcpp;

void sender() {
	PlotServer& server = PlotServer::GetInstance();

	for (int i = 0; i < 10; i ++) {
		server.newEvent(std::make_pair(TrainLossIndex, 2.0));
	}
}

void receiver() {
	std::thread t(PlotServer::Run);
	t.join();
}

int main() {
	sender();

//	PlotServer::GetInstance().test();
	sleep(1);
	receiver();
	sleep(10);

//	sender();
}
