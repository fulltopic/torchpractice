/*
 * clientconn.cpp
 *
 *  Created on: Oct 28, 2020
 *      Author: zf
 */

#include "selfserver/clientconn.h"
#include "selfserver/room.h"
#include "utils/logger.h"

#include "tenhouclient/fsmtypes.h"
#include "utils/logger.h"

#include <string>

#include <boost/algorithm/string.hpp>
#include <boost/lockfree/spsc_queue.hpp>

//TODO: TO use lock-less memory queue for async-sending

using S = std::string;

namespace {
	auto logger = Logger::GetLogger();
}

ClientConn::ClientConn(int playerIndex, boost::asio::io_context& ioService, std::shared_ptr<Room> iRoom)
	:room(iRoom),
	index(playerIndex),
	skt(ioService),
	testTimer(ioService)
{
	logger->info("Client connection {} for Room {} created", index, iRoom->seq);

	for (int i = 0 ; i < WrtBufCap; i ++) {
		wrtBufIndice.push(i);
	}
}

std::shared_ptr<ClientConn> ClientConn::Create(int playerIndex, boost::asio::io_context& ioService, std::shared_ptr<Room> iRoom) {
	return std::shared_ptr<ClientConn>(new ClientConn(playerIndex, ioService, iRoom));
}

void ClientConn::handleSend(const boost::system::error_code& e, std::size_t len, int bufIndex, std::size_t expLen) {
	if (!e) {
		logger->info("Client{}:{} push buffer {}: {}?{}", room->seq, index, bufIndex, len, expLen);
		if (len > 0) {
			//nothing
		}
	} else {
		logger->error("Client {}:{} sending failure: {} ", room->seq, index, e.message());
	}
	wrtBufIndice.push(bufIndex);
}

void ClientConn::start() {
	logger->info("Client {}:{} to start", room->seq, index);
	room->addClient(index, shared_from_this());
	rcv();
	logger->info("Client {}:{} started", room->seq, index);

	testTimer.expires_from_now(boost::posix_time::seconds(10));
	testTimer.async_wait(boost::bind(&ClientConn::testTimerHandle, this->shared_from_this(),
								boost::asio::placeholders::error));

}

//TODO: infinite loop
void ClientConn::send(std::string& msg) {
	this->send(std::move(msg));
}

void ClientConn::send(std::string&& msg) {
	if(msg.length() == 0) {
		return;
	}
	if (msg.compare(StateReturnType::Nothing) == 0) {
		return;
	}

	if (msg.find(StateReturnType::SplitToken) != S::npos) {
		boost::erase_all(msg, StateReturnType::SplitToken);
	}
	msg.resize(msg.length() + 1);
	logger->debug("Client {}:{} to send message {}", room->seq, index, msg);

	int bufIndex;
	int bufSize = msg.length();
	wrtBufIndice.pop(bufIndex);
	std::copy(msg.begin(), msg.end(), wrtBufs[bufIndex]);
	logger->info("Client{}:{} get buffer {}:{}", room->seq, index, bufIndex, msg.length());

    boost::asio::async_write(skt, boost::asio::buffer(wrtBufs[bufIndex], msg.length()),
        boost::bind(&ClientConn::handleSend, shared_from_this(),
          boost::asio::placeholders::error,
          boost::asio::placeholders::bytes_transferred,
		  bufIndex,
		  bufSize));
}

void ClientConn::rcv() {
	skt.async_receive(
	boost::asio::buffer(rcvBuf),
	boost::bind(&ClientConn::handleRcv, this->shared_from_this(),
			boost::asio::placeholders::error,
			boost::asio::placeholders::bytes_transferred())
	);
}

void ClientConn::handleRcv(const boost::system::error_code& e, std::size_t len) {
//	logger->info("handleRcv {}", len);
	if ((!e) || (e == boost::asio::error::message_size)) {
		S msg (rcvBuf.data(), len);

//		if (msg.find("HELO") != S::npos) {
//			logger->warn("Client {} --> {} received helo {}", (void*)this, index, msg);
//		}

		room->processMsg(index, msg);

		if (room->isWorking()) {
			rcv();//rcvBuf is thread-safe
		} else {
			logger->warn("Client {}:{} not working", room->seq, index);
		}
	} else {
		logger->error("Client {}:{} failed to handleRcv: {}", room->seq, index, e.message());
	}
}



void ClientConn::close() {
	//TODO: Should put socket close here? or destructor? Or socket object would be closed automatically?
}

ClientConn::~ClientConn() {
	skt.close();
}

void ClientConn::testTimerHandle(const boost::system::error_code& e) {
	if (!e) {
		send(std::to_string(index) + "<TESTTESTTEST/>");
		testTimer.expires_from_now(boost::posix_time::seconds(10));
		testTimer.async_wait(boost::bind(&ClientConn::testTimerHandle, this->shared_from_this(),
									boost::asio::placeholders::error));
		logger->info("Conn{}:{} send test", room->seq, index);
	} else {
		logger->error("Timer error: {}", e.message());
	}
}
