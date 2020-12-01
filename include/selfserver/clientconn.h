/*
 * clientconn.h
 *
 *  Created on: Oct 27, 2020
 *      Author: zf
 */

#ifndef INCLUDE_SELFSERVER_CLIENTCONN_H_
#define INCLUDE_SELFSERVER_CLIENTCONN_H_


#include <ctime>
#include <iostream>
#include <string>
#include <boost/bind/bind.hpp>
#include <boost/asio.hpp>
#include <boost/array.hpp>

//#include "utils/dataqueue.hpp"
#include <boost/lockfree/queue.hpp>
#include <boost/lockfree/spsc_queue.hpp>

//#include "room.h"
class Room;

//using socket = boost::asio::ip::tcp::socket;


class ClientConn: public std::enable_shared_from_this<ClientConn> {
private:
	enum {RcvBufSize = 512, WrtBufCap = 8};

	std::weak_ptr<Room> room;
	boost::asio::ip::tcp::socket skt;
	boost::array<char, RcvBufSize> rcvBuf;

	boost::lockfree::spsc_queue<int, boost::lockfree::capacity<WrtBufCap>> wrtBufIndice;
	boost::array<char[RcvBufSize], 8> wrtBufs;

	int index;
	int roomIndex;

	ClientConn(int playerIndex, boost::asio::io_context& ioService, std::shared_ptr<Room>& iRoom);

	void handleRcv(const boost::system::error_code& e, std::size_t len);
	void handleSend(const boost::system::error_code& e, std::size_t len, int bufIndex, std::size_t expLen);
	//TODO: Should put socket.close in destructor?

	boost::asio::deadline_timer testTimer;
	void testTimerHandle(const boost::system::error_code& e);

	ClientConn(const ClientConn& ) = delete;
	ClientConn& operator= (const ClientConn&) = delete;
	ClientConn(const ClientConn&&) = delete;
	ClientConn& operator= (const ClientConn&&) = delete;

public:
	~ClientConn();

	static std::shared_ptr<ClientConn> Create(int playerIndex, boost::asio::io_context& ioService, std::shared_ptr<Room>& iRoom);

	void start();

//	void send(std::string& msg);
//	void send(std::string&& msg);
	//TODO: &&
	void send(const std::string& msg);
	void rcv();

	inline boost::asio::ip::tcp::socket& getSocket() {
		return skt;
	}

	void close();
};

#endif /* INCLUDE_SELFSERVER_CLIENTCONN_H_ */
