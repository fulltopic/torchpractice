/*
 * LmdbReaderWrapper.h
 *
 *  Created on: Feb 22, 2020
 *      Author: zf
 */

#ifndef INCLUDE_LMDBTOOLS_LMDBREADERWRAPPER_H_
#define INCLUDE_LMDBTOOLS_LMDBREADERWRAPPER_H_

#include "LmdbReader.h"
#include <queue>
#include <vector>
#include <thread>
#include <condition_variable>
#include <mutex>
#include <torch/torch.h>
#include <iostream>

namespace {
using DBDataType = std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>>;
}

template<typename DataDefs>
class ReaderWrapper {
private:
	LmdbSceneReader<DataDefs> reader;
	const int batchSize;
	const int seqLen;
	const int qCap;
	std::queue<std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>>> q;
	std::mutex wLock;
	std::condition_variable wC;
	bool isRunning;
	bool toReset;
//	std::mutex rLock;
//	std::condition_variable rC;
//
//	void wWait();
//	void wNotify();
//	void rWait();
//	void rNotify();


public:
	ReaderWrapper(std::string path, const int bSize, const int sLen, const int cap);
	~ReaderWrapper() = default;
	void reset();
	void start();
	void stop();
	std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> read();
	std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> getValidSet(const int startPos, const int num);
//		reader(path), batchSize(bSize), seqLen(sLen);
};

template<typename DataDefs>
ReaderWrapper<DataDefs>::ReaderWrapper(std::string path, const int bSize, const int sLen, const int cap):
		reader(path),
		batchSize(bSize),
		seqLen(sLen),
		qCap(cap),
		isRunning(true),
		toReset(false)
		{

}

//template<typename DataDefs>
//void ReaderWrapper<DataDefs>::wWait() {
//	std::unique_lock<std::mutex> lock(wLock);
//	wC.wait(lock);
//}
//
//template<typename DataDefs>
//void ReaderWrapper<DataDefs>::rWait() {
//	std::unique_lock<std::mutex> lock(rLock);
//	rC.wait(lock);
//}
//
////TODO: To be locked?
//template<typename DataDefs>
//void ReaderWrapper<DataDefs>::wNotify() {
//	std::unique_lock<std::mutex> lock(wLock);
//	wC.notify_all();
//}
//
//template<typename DataDefs>
//void ReaderWrapper<DataDefs>::rNotify() {
//	std::unique_lock<std::mutex> lock(rLock);
//	rC.notify_all();
//}
template<typename DataDefs>
DBDataType ReaderWrapper<DataDefs>::getValidSet(const int startPos, const int num) {
	int pos = 0;
	reader.reset();
	while (reader.hasNext() && pos < startPos) {
		reader.next();
		pos ++;
	}

//	std::vector<torch::Tensor> inputs;
	std::vector<torch::Tensor> labels;
	return std::move(reader.next(num, seqLen));
}

template<typename DataDefs>
void ReaderWrapper<DataDefs>::reset() {
	toReset = true;
}

template<typename DataDefs>
void ReaderWrapper<DataDefs>::stop() {
	isRunning = false;
}

template<typename DataDefs>
void ReaderWrapper<DataDefs>::start() {
	while(isRunning) {
		while (q.size() < qCap) {
			if (toReset) {
				reader.reset();
				toReset = false;
			}
			auto data = reader.next(batchSize, seqLen);
			std::cout << "Read data " << std::endl;
			wLock.lock();
			q.push(data);
			wLock.unlock();
		}

		if(q.size() >= qCap) {
			std::unique_lock<std::mutex> uLock(wLock);
			wC.wait(uLock);
		}
	}
}

template<typename DataDefs>
std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> ReaderWrapper<DataDefs>::read() {
	while (q.empty()) {
		sleep(1);
	}

	wLock.lock();
	auto data = q.front();
	q.pop();
	if (q.size() <= (qCap / 2)) {
		wC.notify_all();
	}
	wLock.unlock();

	return std::move(data);
}

#endif /* INCLUDE_LMDBTOOLS_LMDBREADERWRAPPER_H_ */
