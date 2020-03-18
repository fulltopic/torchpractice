/*
 * plotserver.h
 *
 *  Created on: Feb 28, 2020
 *      Author: zf
 */

#ifndef INCLUDE_PYTOOLS_PLOTSERVER_H_
#define INCLUDE_PYTOOLS_PLOTSERVER_H_

#include <vector>
#include <queue>
#include <atomic>
#include <time.h>

enum PlotServerConst {
	FigureW = 2000,
	FigureH = 2000,

	PlotW = 2,
	PlotH = 2,
	PlotNum = PlotW * PlotH,

	TrainLossIndex = 1,
	TrainAccuIndex = 2,
	ValidLossIndex = 3,
	ValidAccuIndex = 4,

	Cap = 1000,
	Threshold = Cap / 2,

	QCap = 128,

	WaitSec = 0,
	WaitMSec = 10 * 1000 * 1000,
};

class PlotServer {
private:
	std::vector<std::vector<float>> datas;
	std::vector<int> indexes;
	std::vector<std::pair<int, float>> q;
	std::atomic_uint qPushIndex;
	std::atomic_uint qPopIndex;
	std::atomic_uint qPushTail;

	const struct timespec waitT;

	bool running;

	void newData(int index, float data);
	void refresh(int index);
	PlotServer();

public:
	~PlotServer() = default;

	PlotServer(const PlotServer& copied) = delete;
	PlotServer& operator=(const PlotServer& copied) = delete;

	void newEvent(std::pair<int, float> data);
	void threadMain();
	void stop();

	void test();

	static PlotServer& GetInstance();
	static void Run();
};


#endif /* INCLUDE_PYTOOLS_PLOTSERVER_H_ */
