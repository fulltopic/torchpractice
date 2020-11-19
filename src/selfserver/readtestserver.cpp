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
boost::asio::ip::tcp::acceptor acceptor(ioService, boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), port));
boost::asio::ip::tcp::socket clientSkt(ioService);

//boost::array<char, 512> rcvBuf;
boost::asio::streambuf rcvBuf;
char wrtBuf[512];
bool accepted = false;


void handleSend(const boost::system::error_code& e, std::size_t len, std::size_t expLen) {
	if (!e) {
		std::cout << "Sent message " << len << " ? " << expLen << std::endl;
		if (len > 0) {
			//nothing
		}
	} else {
		std::cout << "Send message failure " << e.message() << std::endl;
	}
}

void send(std::string msg) {
	msg.resize(msg.length() + 1);
	std::cout << "To send message " << msg << std::endl;
	std::copy(msg.begin(), msg.end(), wrtBuf);

    boost::asio::async_write(clientSkt, boost::asio::buffer(wrtBuf, msg.length()),
        boost::bind(&handleSend,
          boost::asio::placeholders::error,
          boost::asio::placeholders::bytes_transferred,
		  msg.length()));
}

void handleRcv(const boost::system::error_code& e, std::size_t len) {
//	logger->info("handleRcv {}", len);
	if ((!e) || (e == boost::asio::error::message_size)) {
		std::cout << "Received message " << std::endl;
	} else {
		std::cout << "Failed to receive message " << e.message() << std::endl;
	}
}

void rcv() {
	boost::asio::async_read_until(clientSkt,
			rcvBuf,
			"/>",
			boost::bind(handleRcv,
					boost::asio::placeholders::error,
					boost::asio::placeholders::bytes_transferred()
			)
	);
}

void handleAccept(const boost::system::error_code& error) {
	if (!error) {
		//nothing
		std::cout << "Accepted client " << std::endl;
		accepted = true;
		rcv();
	} else {
		std::cout << "Accept error: " << error.message() << std::endl;
	}
}

void accept() {
	acceptor.async_accept(clientSkt,
	        boost::bind(&handleAccept,
	          boost::asio::placeholders::error));
	std::cout << "Ready to accept" << std::endl;
}

void start() {
	accept();
}

void testSend() {
	while (!accepted) {
		sleep(1);
	}
	send("<TESTTESTTEST0/><TESTTESTTEST1/><TESTTESTTEST2/><TESTTESTTEST3/>");
}

void test() {
	start();

	std::vector<std::unique_ptr<std::thread>> ioThreads;
	for (int i = 0; i < 1; i ++) {
		ioThreads.push_back(std::make_unique<std::thread>(
				static_cast<std::size_t (boost::asio::io_context::*)()>(&boost::asio::io_context::run), &ioService));
	}

	testSend();

	for (int i = 0; i < 4; i ++) {
		ioThreads[i]->join();
	}
}
}

int main(int argc, char** argv) {
	test();
}

