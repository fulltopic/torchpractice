/*
 * dataqueue.h
 *
 *  Created on: Sep 16, 2020
 *      Author: zf
 */

#ifndef INCLUDE_UTILS_DATAQUEUE_HPP_
#define INCLUDE_UTILS_DATAQUEUE_HPP_

#include <atomic>
#include <vector>
#include <set>

#include <unistd.h>

template <typename DataType, int Capcity>
class R1WmQueue {
private:
	std::atomic_uint writeSeq; // Cell had been written
	std::atomic_uint holdSeq; // Cell to be written
	std::atomic_uint readSeq; // Cell could be read

	std::vector<DataType> datas;

//	std::set<uint32_t> tmpHolds; //TODO: Make it atomic

public:
	R1WmQueue();
	~R1WmQueue();

	R1WmQueue(const R1WmQueue& other) = delete;
	//TODO: other constructors;

	bool push(DataType&& data);
	DataType pop(); //TODO: Could it be DataType&&?

	bool isEmpty();

	static int GetCapacity();
};

template <typename DataType, int Capacity>
R1WmQueue<DataType, Capacity>::R1WmQueue(): writeSeq((uint32_t)(-1)), holdSeq(0), readSeq(0)
//	, datas(Capacity)
{
}

template<typename DataType, int Capacity>
R1WmQueue<DataType, Capacity>::~R1WmQueue() {

}

template<typename DataType, int Capacity>
bool R1WmQueue<DataType, Capacity>::push(DataType&& data) {
	uint32_t seq = holdSeq;
	uint32_t nextSeq = seq + 1;
	bool toGetSeq = true;
	while (toGetSeq) {
		if ((nextSeq - readSeq) >= Capacity) {
			return false; //full
		}

		toGetSeq = !(holdSeq.compare_exchange_weak(seq, nextSeq));
		if (toGetSeq) {
			seq = holdSeq;
			nextSeq = seq + 1;
		}
	}

	uint32_t index = nextSeq % Capacity;
	datas[index] = data;
	while ((writeSeq + 1) != seq) {
		sleep(1);
	}
	writeSeq = seq;

	return true;
}

template<typename DataType, int Capacity>
DataType R1WmQueue<DataType, Capacity>::pop() {
	uint32_t index = readSeq % Capacity;
	DataType data = datas[index];

	readSeq ++;
	return data;
}

template<typename DataType, int Capacity>
bool R1WmQueue<DataType, Capacity>::isEmpty() {
	if ((writeSeq - readSeq) >= Capacity) {
		return true;
	}else {
		return false;
	}
}

template<typename DataType, int Capacity>
int R1WmQueue<DataType, Capacity>::GetCapacity() {
	return Capacity;
}


#endif /* INCLUDE_UTILS_DATAQUEUE_HPP_ */
