/*
 * batchplotserver.h
 *
 *  Created on: Mar 3, 2020
 *      Author: zf
 */

#ifndef INCLUDE_PYTOOLS_BATCHPLOTSERVER_H_
#define INCLUDE_PYTOOLS_BATCHPLOTSERVER_H_

#include <vector>
#include <queue>
#include <atomic>
#include <time.h>
#include <condition_variable>
#include <mutex>

#include "plotserver.h"

class BatchPlotServer {
private:
	std::vector<std::vector<float>> datas;
	std::vector<int> indexes;
	std::vector<std::pair<int, float>> q;
	volatile bool toRead;
	int pushIndex;
//	std::atomic_uint qPushIndex;
//	std::atomic_uint qPopIndex;
//	std::atomic_uint qPushTail;

	const struct timespec waitT;

	bool running;

	std::mutex pushMutex;
	std::mutex popMutex;
	std::condition_variable pushCv;
	std::condition_variable popCv;

	void newData(int index, float data);
	void refresh();
	BatchPlotServer();

public:
	~BatchPlotServer() = default;

	BatchPlotServer(const BatchPlotServer& copied) = delete;
	BatchPlotServer& operator=(const BatchPlotServer& copied) = delete;

	void newEvent(std::pair<int, float> data);
	void threadMain();
	inline void stop() { running = false; }
	void notifyRead();

	void test();

	static BatchPlotServer& GetInstance();
	static void Run();
};





#endif /* INCLUDE_PYTOOLS_BATCHPLOTSERVER_H_ */
