/*
 * selfserver.h
 *
 *  Created on: Oct 27, 2020
 *      Author: zf
 */

#ifndef INCLUDE_SELFSERVER_SELFSERVER_H_
#define INCLUDE_SELFSERVER_SELFSERVER_H_

#include <map>
#include <ctime>
#include <iostream>
#include <string>
#include <mutex>

#include <boost/bind/bind.hpp>
#include <boost/asio.hpp>

#include "room.h"
#include "clientconn.h"

enum class ServerConfig {
	CleanUpTimeout = 60 * 2,
};

class SelfServer:
		public std::enable_shared_from_this<SelfServer>  {
private:
	std::map<uint32_t, std::shared_ptr<Room>> rooms;
	uint32_t roomSeq = 0;

	const int port;
	std::mutex roomMutex;
	std::shared_ptr<Room> curRoom;
	int curClientIndex = 0;

	boost::asio::io_context& ioService;
	boost::asio::ip::tcp::acceptor acceptor;
	boost::asio::deadline_timer pollTimer;

	void handleAccept(std::shared_ptr<ClientConn> client, const boost::system::error_code& error);
	void handleRoomPoll(const boost::system::error_code& error);

	std::pair<std::shared_ptr<Room>, int> getRoomIndex();

public:
	SelfServer(int iPort, boost::asio::io_context& service);
	void start();
	void accept();
};



#endif /* INCLUDE_SELFSERVER_SELFSERVER_H_ */
