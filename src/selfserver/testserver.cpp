/*
 * testserver.cpp
 *
 *  Created on: Nov 6, 2020
 *      Author: zf
 */



#include "selfserver/selfserver.h"
#include "utils/logger.h"
#include "spdlog/spdlog.h"
#include <boost/asio.hpp>
#include <boost/thread/thread.hpp>
#include <boost/bind/bind.hpp>
#include <thread>


namespace {
void startServer() {
	spdlog::set_level(spdlog::level::info);
//	Logger::GetLogger()->set_pattern("[%t][%l]: %v");

	boost::asio::io_context io;

	std::shared_ptr<SelfServer> server = std::make_shared<SelfServer>(55555, io);
	server->start();

	std::vector<std::unique_ptr<std::thread>> ioThreads;
	for (int i = 0; i < 4; i ++) {
		ioThreads.push_back(std::make_unique<std::thread>(
				static_cast<std::size_t (boost::asio::io_context::*)()>(&boost::asio::io_context::run), &io));
	}

	for (int i = 0; i < 4; i ++) {
		ioThreads[i]->join();
	}

}
}

int main() {
	startServer();
}
