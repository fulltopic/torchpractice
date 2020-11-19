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

ClientConn::ClientConn(int playerIndex, boost::asio::io_context& ioService, std::shared_ptr<Room>& iRoom)
	:room(iRoom),
	index(playerIndex),
	roomIndex(iRoom->seq),
	skt(ioService),
	testTimer(ioService)
{
	logger->info("Client connection {} for Room {} created", index, iRoom->seq);

	for (int i = 0 ; i < WrtBufCap; i ++) {
		wrtBufIndice.push(i);
	}
}

std::shared_ptr<ClientConn> ClientConn::Create(int playerIndex, boost::asio::io_context& ioService, std::shared_ptr<Room>& iRoom) {
	return std::shared_ptr<ClientConn>(new ClientConn(playerIndex, ioService, iRoom));
}

void ClientConn::handleSend(const boost::system::error_code& e, std::size_t len, int bufIndex, std::size_t expLen) {
	if ((!e) || (e == boost::asio::error::message_size)) {
//		logger->info("Client{}:{} push buffer {}: {}?{}", room->seq, index, bufIndex, len, expLen);
		if (len > 0) {
			//nothing
		}
		wrtBufIndice.push(bufIndex);
	} else {
		logger->error("Client {}:{} sending failure: {} ", roomIndex, index, e.message());
		auto roomPtr = room.lock();
		if (roomPtr) {
			roomPtr->processNetErr(index);
		}
	}
}

void ClientConn::start() {
	logger->info("Client {}:{} to start", roomIndex, index);
	room.lock()->addClient(index, shared_from_this());
	rcv();
	logger->info("Client {}:{} started", roomIndex, index);
//
//	testTimer.expires_from_now(boost::posix_time::seconds(10));
//	testTimer.async_wait(boost::bind(&ClientConn::testTimerHandle, this->shared_from_this(),
//								boost::asio::placeholders::error));

}

//TODO: infinite loop
//void ClientConn::send(std::string& msg) {
//	this->send(std::move(msg));
//}

void ClientConn::send(std::string msg) {
//void ClientConn::send(std::string&& msg) {
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
	logger->info("Client {}:{} to send message {}", roomIndex, index, msg);

	int bufIndex;
	int bufSize = msg.length();
	while (!wrtBufIndice.pop(bufIndex)) {
		logger->error("Client{}:{} Not enough buffer to be written", roomIndex, index);
	}
	std::copy(msg.begin(), msg.end(), wrtBufs[bufIndex]);
//	logger->info("Client{}:{} get buffer {}:{}", room->seq, index, bufIndex, msg.length());

    boost::asio::async_write(skt, boost::asio::buffer(wrtBufs[bufIndex], msg.length()),
        boost::bind(&ClientConn::handleSend, this->shared_from_this(),
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
	auto roomPtr = room.lock();
	if ((!e) || (e == boost::asio::error::message_size)) {
		S msg (rcvBuf.data(), len);

//		if (msg.find("HELO") != S::npos) {
//			logger->warn("Client {} --> {} received helo {}", (void*)this, index, msg);
//		}

		roomPtr->processMsg(index, msg);

		if (roomPtr->isWorking()) {
			rcv();//rcvBuf is thread-safe
		} else {
			logger->warn("Client {}:{} not working", roomIndex, index);
		}
	} else {
		logger->error("Client {}:{} failed to handleRcv: {}", roomIndex, index, e.message());
		if (roomPtr) {
			roomPtr->processNetErr(index);
		} else {
			logger->error("Room{} had been closed", roomIndex);
		}
	}
}



void ClientConn::close() {
	//TODO: Should put socket close here? or destructor? Or socket object would be closed automatically?
	try {
		skt.cancel();
		testTimer.cancel();
	} catch (std::exception& e) {
		logger->error("Client{}:{} Failed to cancel socket {}", roomIndex, index, e.what());
	}
}

ClientConn::~ClientConn() {
	try {
	skt.close();
	logger->warn("Client {}: {} destructed", roomIndex, index);
	} catch(std::exception& e) {
		logger->error("Client{}:{} Failed to close socket as {}", roomIndex, index, e.what());
	}
}

void ClientConn::testTimerHandle(const boost::system::error_code& e) {
	if (!e) {
		send(std::to_string(index) + "<TESTTESTTEST/>");
		testTimer.expires_from_now(boost::posix_time::seconds(10));
		testTimer.async_wait(boost::bind(&ClientConn::testTimerHandle, this->shared_from_this(),
									boost::asio::placeholders::error));
		logger->info("Conn{}:{} send test", roomIndex, index);
	} else {
		logger->error("Timer error: {}", e.message());
	}
}
