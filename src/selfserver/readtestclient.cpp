/*
 * readtestserver.cpp
 *
 *  Created on: Nov 12, 2020
 *      Author: zf
 */

#include <map>
#include <ctime>
#include <iostream>
#include <string>
#include <mutex>
#include <memory>
#include <thread>

#include <boost/bind/bind.hpp>
#include <boost/asio.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/array.hpp>

namespace {
int port = 25324;

boost::asio::io_context ioService;
boost::asio::ip::tcp::endpoint serverP (
		boost::asio::ip::address::from_string("127.0.0.1"), port);

boost::asio::ip::tcp::socket skt(ioService);

//boost::array<char, 512> rcvBuf;
boost::asio::streambuf rcvBuf(512);
boost::array<char, 512> wrtBuf;

void rcv();

void handleRcv(const boost::system::error_code& e, std::size_t len) {
//	logger->info("handleRcv {}", len);
	if ((!e) || (e == boost::asio::error::message_size)) {
		std::cout << "Received " << len << std::endl;
		std::vector<char> msgVec(len);
		std::istream tmpStr(&rcvBuf);
		tmpStr.read(&msgVec[0], len);
		std::string msg(&msgVec[0], &msgVec[len]);
//		rcvBuf.consume(len);
//		rcvBuf.commit(len);
//		std::string msg;
//		tmpStr >> msg;
		std::cout << "Received message " << msg << std::endl;
		rcv();
	} else {
		std::cout << "Failed to receive message " << e.message() << std::endl;
	}
}

void rcv() {
	boost::asio::async_read_until(skt,
			rcvBuf,
			'>',
			boost::bind(handleRcv,
					boost::asio::placeholders::error,
					boost::asio::placeholders::bytes_transferred()
			)
	);
}

void start() {
	skt.open(boost::asio::ip::tcp::v4());
	skt.connect(serverP);

	rcv();
}

void test() {
	start();

	std::vector<std::unique_ptr<std::thread>> ioThreads;
	for (int i = 0; i < 1; i ++) {
		ioThreads.push_back(std::make_unique<std::thread>(
				static_cast<std::size_t (boost::asio::io_context::*)()>(&boost::asio::io_context::run), &ioService));
	}

	for (int i = 0; i < 1; i ++) {
		ioThreads[i]->join();
	}
}
}

int main() {
	test();
}
